from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext as _
from .forms import RegistrationForm, MessageForm
from .models import Conversation, Message

def _ensure_conversation(request, conv_id=None):
    if conv_id:
        return get_object_or_404(Conversation, pk=conv_id)
    # Show latest for user, or create a guest conversation
    if request.user.is_authenticated:
        conv = request.user.conversations.first()
        if conv:
            return conv
        return Conversation.objects.create(user=request.user, title=_("Cuộc trò chuyện mới"))
    # Anonymous fallback: a session-scoped conversation id
    if "guest_conv_id" in request.session:
        try:
            return Conversation.objects.get(pk=request.session["guest_conv_id"])
        except Conversation.DoesNotExist:
            pass
    conv = Conversation.objects.create(user=None, title=_("Cuộc trò chuyện (Khách)"))
    request.session["guest_conv_id"] = conv.id
    return conv

def index(request, pk=None):
    conv = _ensure_conversation(request, pk)
    convs = Conversation.objects.filter(user=request.user) if request.user.is_authenticated else Conversation.objects.filter(pk=conv.pk)
    form = MessageForm()
    return render(
        request,
        "chat/index.html",
        {
            "conversations": convs,
            "active_conversation": conv,
            "messages": conv.messages.all(),
            "form": form,
        },
    )

def new_conversation(request):
    if request.user.is_authenticated:
        conv = Conversation.objects.create(user=request.user, title=_("Cuộc trò chuyện mới"))
    else:
        conv = Conversation.objects.create(user=None, title=_("Cuộc trò chuyện (Khách)"))
        request.session["guest_conv_id"] = conv.id
    return redirect("chat:conversation", pk=conv.pk)

def send_message(request, pk):
    conv = get_object_or_404(Conversation, pk=pk)
    if request.method != "POST":
        return redirect("chat:conversation", pk=pk)
    form = MessageForm(request.POST, request.FILES)
    if form.is_valid():
        msg = form.save(commit=False)
        msg.conversation = conv
        msg.sender = "user"
        msg.save()
        # Simple bot echo for demo
        Message.objects.create(
            conversation=conv,
            sender="bot",
            content=_("Cảm ơn bạn! Tôi đã nhận: ") + (msg.content or _("(tệp đính kèm)")),
        )
        return redirect("chat:conversation", pk=pk)
    # On error, re-render
    convs = Conversation.objects.filter(user=request.user) if request.user.is_authenticated else Conversation.objects.filter(pk=conv.pk)
    return render(
        request,
        "chat/index.html",
        {
            "conversations": convs,
            "active_conversation": conv,
            "messages": conv.messages.all(),
            "form": form,
        },
    )

def register(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, _("Đăng ký thành công."))
            return redirect("chat:index")
    else:
        form = RegistrationForm()
    # Render base with modal open state
    return render(request, "chat/register_page.html", {"form": form})