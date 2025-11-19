
import torch.nn as nn
import torch.nn.functional as F

class TeacherStudentModel(nn.Module):
    """
    A wrapper module for performing knowledge distillation in a teacher-student setup.

    The student model learns from the teacher model by minimizing a combined loss
    that includes both the original task loss and a distillation loss.
    """
    def __init__(self, student_model, teacher_model):
        """
        Initializes the TeacherStudentModel.

        Args:
            student_model (nn.Module): The model that is being trained (student).
            teacher_model (nn.Module): The pre-trained model providing knowledge (teacher).
        """
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model

    def forward(self, *args, **kwargs):
        """
        Performs a forward pass, calculates the student's loss, and adds distillation loss.

        Args:
            *args: Positional arguments passed to both student and teacher models.
            **kwargs: Keyword arguments passed to both student and teacher models.

        Returns:
            object: The output of the student model with the modified loss.
        """
        student_output = self.student_model(*args, **kwargs)
        
        with torch.no_grad():
            teacher_output = self.teacher_model(*args, **kwargs)

        # Add distillation loss
        # Assuming student_output.logits and teacher_output.logits are available
        # and have compatible shapes for KL divergence
        distillation_loss = F.kl_div(
            F.log_softmax(student_output.logits / 2.0, dim=-1),
            F.softmax(teacher_output.logits / 2.0, dim=-1),
            reduction='batchmean'
        ) * (2.0 ** 2)

        # Combine with original loss
        original_loss = student_output.loss
        loss = 0.5 * original_loss + 0.5 * distillation_loss
        
        student_output.loss = loss
        return student_output
