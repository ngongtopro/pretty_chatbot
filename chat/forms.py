from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Message

class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

class MessageForm(forms.ModelForm):
    content = forms.CharField(
        label="",
        widget=forms.Textarea(
            attrs={
                "rows": 3,
                "placeholder": "Nhập tin nhắn...",
                "class": "w-full resize-none rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 p-3 focus:outline-none focus:ring-2 focus:ring-emerald-500",
            }
        ),
        required=False,
    )

    class Meta:
        model = Message
        fields = ("content", "attachment")

    def clean(self):
        cleaned = super().clean()
        if not cleaned.get("content") and not cleaned.get("attachment"):
            raise forms.ValidationError("Vui lòng nhập nội dung hoặc đính kèm tệp.")
        return cleaned