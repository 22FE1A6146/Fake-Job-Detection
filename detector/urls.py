from django.urls import path
from .views import predict_job, send_chat_message

urlpatterns = [
    path('', predict_job, name='predict_job'),
    path('predict/', predict_job, name='predict_job'),
    path('chat/send/', send_chat_message, name='send_chat_message'),
]