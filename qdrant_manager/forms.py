from django import forms
from .models import QdrantCollection, QdrantPoint

class QdrantCollectionForm(forms.ModelForm):
    class Meta:
        model = QdrantCollection
        fields = ['name', 'description', 'vector_size', 'distance_metric']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500',
                'placeholder': 'Nhập tên collection...'
            }),
            'description': forms.Textarea(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500',
                'rows': 3,
                'placeholder': 'Nhập mô tả...'
            }),
            'vector_size': forms.NumberInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500',
                'placeholder': '1536',
                'min': '1',
                'max': '2048'
            }),
            'distance_metric': forms.Select(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500'
            })
        }

class QdrantPointForm(forms.ModelForm):
    generate_vector = forms.BooleanField(
        required=False,
        initial=True,
        label="Tự động tạo vector",
        help_text="Tích vào đây để hệ thống tự động tạo vector từ nội dung"
    )
    
    class Meta:
        model = QdrantPoint
        fields = ['point_id', 'content', 'title', 'source', 'file_name', 'category', 'tags']
        widgets = {
            'point_id': forms.TextInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500',
                'placeholder': 'Nhập Point ID...'
            }),
            'content': forms.Textarea(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500',
                'rows': 5,
                'placeholder': 'Nhập nội dung...'
            }),
            'title': forms.TextInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500',
                'placeholder': 'Nhập tiêu đề...'
            }),
            'source': forms.URLInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500',
                'placeholder': 'https://example.com'
            }),
            'file_name': forms.TextInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500',
                'placeholder': 'Nhập tên file...'
            }),
            'category': forms.TextInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500',
                'placeholder': 'Nhập danh mục...'
            }),
            'tags': forms.TextInput(attrs={
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500',
                'placeholder': 'tag1, tag2, tag3...'
            })
        }
    
    def clean_tags(self):
        """Chuyển đổi tags từ string thành list"""
        tags = self.cleaned_data.get('tags', '')
        if isinstance(tags, str):
            return [tag.strip() for tag in tags.split(',') if tag.strip()]
        return tags

class BulkDeleteForm(forms.Form):
    """Form để xóa nhiều points cùng lúc"""
    point_ids = forms.CharField(widget=forms.HiddenInput())

class SyncCollectionForm(forms.Form):
    """Form để sync collection với Qdrant server"""
    collection_name = forms.CharField(widget=forms.HiddenInput())

class SearchForm(forms.Form):
    """Form tìm kiếm points"""
    query = forms.CharField(
        max_length=500,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500',
            'placeholder': 'Nhập nội dung tìm kiếm...'
        })
    )
    limit = forms.IntegerField(
        initial=10,
        min_value=1,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-emerald-500'
        })
    )
