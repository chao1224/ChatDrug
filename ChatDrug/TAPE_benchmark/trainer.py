import collections
import warnings
from typing import Tuple, Optional, Union, Dict, Any, List

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset, DataLoader
from transformers import Trainer, EvalPrediction, is_torch_tpu_available
from transformers.trainer_pt_utils import find_batch_size, nested_numpify
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize, PredictionOutput
import numpy as np


class OntoProteinTrainer(Trainer):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: False,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            #loss tensor (batch_size, loss)
            #dict: outputs['loss'] (batch_size, loss)
            #outputs['logits'] (batch_size, protein_length, protein_length, num_labels)
            #outputs['prediction_score'] dict{'precision_at_l5:' (batch_size, prediction_score)}
            loss = loss.mean().detach()
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items())
                #logits: Tuple:
                #logits[0] : model_output (batch_size, protein_length, hidden_size)
                #logits[1] : prediction (batch_size, protein_length, protein_length, num_labels)
                #logits[2] : dict{'precision_at_l5:' (batch_size, prediction_score)}
            else:
                logits = outputs[1:]

        if prediction_loss_only:
            pass
            #return (loss, None, None, None)

        logit = logits[2]

        prediction_score = {}

        prediction_score['precision_at_l5'] = logits[3]['precision_at_l5']
        prediction_score['precision_at_l2'] = logits[3]['precision_at_l2']
        prediction_score['precision_at_l'] = logits[3]['precision_at_l']
        labels = inputs['labels']
        if len(logits) == 1:
            logit = logits[0]

        return (loss, logit, labels, prediction_score)

    def prediction_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None):
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        print("***** Running %s *****", description)
        print("  Num examples = %d", num_examples)
        print("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None
        #preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        #labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = 1
        if is_torch_tpu_available():
            world_size = xm.xrt_world_size()
        elif self.args.local_rank != -1:
            world_size = torch.distributed.get_world_size()
        world_size = max(1, world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        #preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
        #labels_gatherer = DistributedTensorGatherer(world_size, num_examples)

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        contact_meterics_l5 = []
        contact_meterics_l2 = []
        contact_meterics_l = []
        for step, inputs in enumerate(dataloader):
            loss, logits, labels, prediction_score = self.prediction_step(model, inputs, prediction_loss_only)
            contact_meterics_l5.append(torch.mean(prediction_score['precision_at_l5']))
            contact_meterics_l2.append(torch.mean(prediction_score['precision_at_l2']))
            contact_meterics_l.append(torch.mean(prediction_score['precision_at_l']))
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))

                # Set back to None to begin a new accumulation
                losses_host = None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        metrics = {}
        eval_loss = eval_losses_gatherer.finalize()
        metrics["accuracy_l5"] = sum(contact_meterics_l5) / len(contact_meterics_l5)
        metrics["accuracy_l2"] = sum(contact_meterics_l2) / len(contact_meterics_l2)
        metrics["accuracy_l"] = sum(contact_meterics_l) / len(contact_meterics_l)
        metrics = denumpify_detensorize(metrics)

        return PredictionOutput(predictions=None, label_ids=None, metrics=metrics)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        print(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            print(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            print("  Num examples: Unknown")
        print(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        losses_host = None

        all_losses = None

        # Will be useful when we have an iterable dataset so don't know its length.
        contact_meterics_l5 = []
        contact_meterics_l2 = []
        contact_meterics_l = []
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            loss, logits, labels, prediction_score = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            contact_meterics_l5.append(torch.mean(prediction_score['precision_at_l5']))
            contact_meterics_l2.append(torch.mean(prediction_score['precision_at_l2']))
            contact_meterics_l.append(torch.mean(prediction_score['precision_at_l']))
            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]

        metrics = {}
        #metrics = prediction_score  # mean
        metrics["accuracy_l5"] = sum(contact_meterics_l5) / len(contact_meterics_l5)
        metrics["accuracy_l2"] = sum(contact_meterics_l2) / len(contact_meterics_l2)
        metrics["accuracy_l"] = sum(contact_meterics_l) / len(contact_meterics_l)
        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)
