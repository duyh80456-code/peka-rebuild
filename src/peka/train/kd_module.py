"""KD trainers for PEKA.

Two variants:
  - ``PekaKDLoRA`` (alias of ``pl_KD_LoRA``)
        Two-phase training. Phase 1 pre-trains the MLP classifier (teacher);
        Phase 2 freezes it and trains the PEFT student against it. Stable
        but adds a ~10-min Phase 1 step.

  - ``PekaKDLoRAJoint``  (paper-aligned)
        Single-stage joint training. Adapter + translate MLP + classifier MLP
        all trainable. Matches paper §4: *"the adapter parameters and MLP
        weights are updated while keeping the backbone frozen"*. Loss is the
        same KL + CE structure-alignment.

The paper itself doesn't require Phase 1 — it's a stability trick from the
original PEKA codebase. ``PekaKDLoRAJoint`` skips it and trains everything
together.
"""
from itertools import chain

from peka.Model.base import MLPClassifier
from peka.Trainer.KD_LoRA import pl_KD_LoRA as PekaKDLoRA  # re-export


class PekaKDLoRAJoint(PekaKDLoRA):
    """Paper-aligned joint training: classifier MLP is trained alongside adapter.

    Differences from ``PekaKDLoRA``:
      1. Classifier is initialized inline in ``__init__`` (no Phase 1 ckpt needed).
      2. Classifier is NOT frozen — its weights update via student-path gradients.
      3. ``configure_optimizers`` includes classifier parameters.
      4. ``setup_teacher_model`` is a warm-start helper (kept for API compatibility).

    Loss structure (unchanged):
        L_total = alpha * KL(student/T || teacher/T) * T**2  +  (1-alpha) * CE(student, label)

    The teacher path inside ``training_step`` uses ``torch.no_grad()`` on the SAME
    classifier instance — KL has a detached "soft target" but the classifier
    still updates from the CE term + KL gradient on the student branch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize classifier inline (replaces pre-trained teacher from Phase 1).
        self.classifier = MLPClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.classifier_hidden_dim,
            num_classes=self.num_classes,
        )
        # Trainable (paper says "MLP weights are updated").
        self.classifier.requires_grad_(True)

    def setup_teacher_model(self, classifier=None):
        """Optional warm-start. In joint training the classifier stays trainable."""
        if classifier is not None:
            self.classifier.load_state_dict(classifier.state_dict())
            self.classifier.requires_grad_(True)

    def configure_optimizers(self):
        """Include classifier params alongside model params in the optimizer."""
        params = [p for p in chain(self.model.parameters(), self.classifier.parameters())
                  if p.requires_grad]
        opt_list = [opt_fn(params) for opt_fn in self.optimizer_instance_list]
        if self.scheduler_instance_list:
            sch_list = [sch_fn(optimizer=opt) for sch_fn, opt in
                        zip(self.scheduler_instance_list, opt_list)]
            return opt_list, sch_list
        return opt_list


__all__ = ["PekaKDLoRA", "PekaKDLoRAJoint"]
