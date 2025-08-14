from django import template
from django.utils.safestring import mark_safe
import json

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Lấy item từ dictionary theo key"""
    if hasattr(dictionary, 'get'):
        return dictionary.get(key)
    return None

@register.filter
def json_pretty(value):
    """Format JSON đẹp hơn"""
    try:
        if isinstance(value, str):
            value = json.loads(value)
        return mark_safe(json.dumps(value, indent=2, ensure_ascii=False))
    except (ValueError, TypeError):
        return value

@register.filter
def truncate_middle(value, max_length=50):
    """Cắt chuỗi ở giữa với dấu ... """
    if not value or len(value) <= max_length:
        return value
    
    if max_length <= 3:
        return "..."
    
    # Chia đôi length để lấy ký tự đầu và cuối
    half = (max_length - 3) // 2
    return f"{value[:half]}...{value[-half:]}"

@register.filter
def multiply(value, multiplier):
    """Nhân hai số"""
    try:
        return float(value) * float(multiplier)
    except (ValueError, TypeError):
        return 0

@register.filter
def percentage(value, total):
    """Tính phần trăm"""
    try:
        if float(total) == 0:
            return 0
        return round((float(value) / float(total)) * 100, 1)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0

@register.filter
def vector_preview(vector_data, max_items=5):
    """Hiển thị preview của vector data"""
    if not vector_data or not isinstance(vector_data, list):
        return "[]"
    
    if len(vector_data) <= max_items:
        formatted = [f"{x:.3f}" for x in vector_data[:max_items]]
        return f"[{', '.join(formatted)}]"
    else:
        formatted = [f"{x:.3f}" for x in vector_data[:max_items]]
        return f"[{', '.join(formatted)}, ...] ({len(vector_data)} dims)"

@register.filter
def similarity_color(score):
    """Trả về màu CSS dựa trên similarity score"""
    try:
        score = float(score)
        if score >= 0.9:
            return "text-green-600"
        elif score >= 0.7:
            return "text-blue-600"
        elif score >= 0.5:
            return "text-yellow-600"
        else:
            return "text-red-600"
    except (ValueError, TypeError):
        return "text-gray-600"

@register.filter
def status_badge(is_synced):
    """Trả về HTML badge cho trạng thái sync"""
    if is_synced:
        return mark_safe(
            '<span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">'
            '<i class="fas fa-check mr-1"></i>Synced</span>'
        )
    else:
        return mark_safe(
            '<span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-yellow-100 text-yellow-800">'
            '<i class="fas fa-clock mr-1"></i>Pending</span>'
        )

@register.filter
def file_size_format(bytes_size):
    """Format file size thành human readable"""
    try:
        bytes_size = int(bytes_size)
        if bytes_size < 1024:
            return f"{bytes_size} B"
        elif bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.1f} KB"
        elif bytes_size < 1024 * 1024 * 1024:
            return f"{bytes_size / (1024 * 1024):.1f} MB"
        else:
            return f"{bytes_size / (1024 * 1024 * 1024):.1f} GB"
    except (ValueError, TypeError):
        return "0 B"

@register.simple_tag
def vector_dimension_badge(size):
    """Tạo badge cho vector dimension"""
    color_map = {
        range(0, 100): "bg-gray-100 text-gray-800",
        range(100, 500): "bg-blue-100 text-blue-800", 
        range(500, 1000): "bg-green-100 text-green-800",
        range(1000, 2000): "bg-yellow-100 text-yellow-800",
    }
    
    color = "bg-red-100 text-red-800"  # default for > 2000
    for size_range, badge_color in color_map.items():
        if size in size_range:
            color = badge_color
            break
    
    return mark_safe(
        f'<span class="px-2 py-1 text-xs font-medium rounded {color}">{size}D</span>'
    )

@register.inclusion_tag('qdrant_manager/components/pagination.html')
def render_pagination(page_obj, search_query=None):
    """Render pagination component"""
    return {
        'page_obj': page_obj,
        'search_query': search_query,
    }

@register.inclusion_tag('qdrant_manager/components/search_form.html')
def render_search_form(search_query=None, placeholder="Tìm kiếm..."):
    """Render search form component"""
    return {
        'search_query': search_query,
        'placeholder': placeholder,
    }
