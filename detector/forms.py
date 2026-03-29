from django import forms

class JobForm(forms.Form):
    title = forms.CharField(
        max_length=300,
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'Job Title'})
    )
    company_profile = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'rows': 4, 'placeholder': 'Company Profile'})
    )
    description = forms.CharField(
        required=True,
        widget=forms.Textarea(attrs={'rows': 6, 'placeholder': 'Job Description'})
    )
    requirements = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'rows': 4, 'placeholder': 'Requirements'})
    )
    benefits = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'rows': 4, 'placeholder': 'Benefits'})
    )