from django.db import models
from django.utils import timezone


class JobPrediction(models.Model):
    """Stores the result of fake job predictions"""
    job_content = models.TextField()
    prediction = models.CharField(max_length=10)   # "Real" or "Fake"
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prediction} ({self.probability:.4f}) - {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class ChatMessage(models.Model):
    """Stores messages from the community chat"""
    username = models.CharField(max_length=100, default="Anonymous")
    message = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']   # Show newest messages first

    def __str__(self):
        return f"{self.username}: {self.message[:60]}"


# Optional: You can add more models later here