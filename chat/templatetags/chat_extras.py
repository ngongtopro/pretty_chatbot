from django import template
from django.utils import timezone

register = template.Library()

@register.filter
def human_timestamp(dt):
    """
    - If same day as today (local time): show HH:MM.
    - Else: show DD/MM/YYYY HH:MM.
    """
    if not dt:
        return ""
    local = timezone.localtime(dt)
    now = timezone.localtime(timezone.now())
    fmt_time = local.strftime("%H:%M")
    if local.date() == now.date():
        return fmt_time
    return local.strftime("%d/%m/%Y ") + fmt_time