from django.shortcuts import render, redirect
from .forms import JobForm
from .ml.model import get_predictor
from .models import JobPrediction, ChatMessage


def predict_job(request):
    """Main dashboard view with dynamic statistics"""
    result = None
    job_content_preview = None

    if request.method == 'POST' and 'check_job' in request.POST:
        form = JobForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            job_content = " ".join([
                data.get('title', ''),
                data.get('company_profile', ''),
                data.get('description', ''),
                data.get('requirements', ''),
                data.get('benefits', '')
            ])

            try:
                predictor = get_predictor()
                result = predictor.predict(job_content)

                JobPrediction.objects.create(
                    job_content=job_content[:1000],
                    prediction=result['prediction'],
    
                )

                job_content_preview = job_content[:300] + "..." if len(job_content) > 300 else job_content
            except Exception as e:
                result = {"error": str(e)}

    else:
        form = JobForm()

    # ================== DYNAMIC STATISTICS ==================
    total_predictions = JobPrediction.objects.count()
    fake_detected = JobPrediction.objects.filter(prediction="Fake").count()
    
    fake_percentage = round((fake_detected / total_predictions * 100), 1) if total_predictions > 0 else 0
    accuracy = round(100 - fake_percentage, 1)   # Simple accuracy approximation

    total_messages = ChatMessage.objects.count()
    # =========================================================

    recent_messages = ChatMessage.objects.all()[:50]

    return render(request, 'predict.html', {
        'form': form,
        'result': result,
        'job_content_preview': job_content_preview,
        'chat_messages': recent_messages,

        # Dynamic values passed to template
        'total_predictions': total_predictions,
        'fake_detected': fake_detected,
        'fake_percentage': fake_percentage,
        'accuracy': accuracy,
        'total_messages': total_messages,
    })


def send_chat_message(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        message = request.POST.get('message', '').strip()
        if message:
            ChatMessage.objects.create(
                username=username if username else "Anonymous",
                message=message
            )
    return redirect('predict_job')