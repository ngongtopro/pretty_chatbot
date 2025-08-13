from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext as _
from django.http import JsonResponse
from .forms import RegistrationForm, MessageForm
from .models import Conversation, Message
import os
from openai import OpenAI


api_key = os.getenv("OPENAI_KEY", None)
if api_key:
    client = OpenAI(api_key=api_key)

# vector_store = client.vector_stores.create(
#     name="chatbot_vector_store"
# )

# client.vector_stores.files.upload_and_poll(
#     vector_store_id=vector_store.id,
#     file=open("data/BO CAU HOI_L1_08082025.txt", "rb")
# )


def get_response_data(msg):
    if not client:
        return msg, 0, 0
    results = client.vector_stores.search(
        vector_store_id="vs_689c184209848191a172344a6c1b1481",
        query=msg.content,
        max_num_results=10
    )
    print("OpenAI search results:\n", results)
    contents = [vector_store_search_response.content[0].text for vector_store_search_response in results]
    combined_content = "\n".join(contents)
    print(combined_content)
    response = client.responses.create(
        model="gpt-3.5-turbo",
        input=[
            {
                "role": "user",
                "content": "\n".join([
                    f"Document: {combined_content}",
                    f"User: {msg.content}"
                ])
            },
            {
                "role": "system", 
                "content": "\n".join([
                    "Bạn là trợ lý AI thân thiện.",
                    "Bạn sẽ trả lời dựa trên thông tin trong tài liệu.",
                    "Nếu câu hỏi không có trong tài liệu, hãy trả lời 'UNKNOWN'.",
                    "Nếu không phải câu hỏi (chào, trò chuyện...), hãy phản hồi tự nhiên, gợi mở câu chuyện.",
                    "Luôn lịch sự, dễ hiểu; không bịa thông tin ngoài tài liệu."
                ])
            }
        ]
    )
    print(response)
    content = response.output[0].content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    return content, input_tokens, output_tokens


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
        try:
            msg = form.save(commit=False)
            msg.conversation = conv
            msg.sender = "user"
            msg.save()
            
            # OpenAI API call
            content, input_tokens, output_tokens = get_response_data(msg)

            # Create bot response
            bot_message = Message.objects.create(
                conversation=conv,
                sender="bot",
                content=f"{content}\n(input tokens: {input_tokens}, output tokens: {output_tokens})",
            )
            
            # Return JSON for AJAX requests
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'bot_message': bot_message.content,
                    'user_message': msg.content
                })
            
            return redirect("chat:conversation", pk=pk)
            
        except Exception as e:
            print(f"Error in send_message: {e}")
            error_message = "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại."
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': False,
                    'error': error_message
                })
            
            return redirect("chat:conversation", pk=pk)
    
    # On error, re-render for non-AJAX requests
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'success': False,
            'error': 'Dữ liệu không hợp lệ.'
        })
    
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
    """Phuc vụ đăng ký người dùng mới"""
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