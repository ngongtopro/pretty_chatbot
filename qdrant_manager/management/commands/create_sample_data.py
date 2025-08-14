from django.core.management.base import BaseCommand
from qdrant_manager.models import QdrantDatabase, QdrantDocument

class Command(BaseCommand):
    help = 'Tạo dữ liệu mẫu cho Qdrant Manager'

    def handle(self, *args, **options):
        self.stdout.write('Tạo dữ liệu mẫu cho Qdrant Manager...')
        
        # Tạo databases mẫu
        db1, created = QdrantDatabase.objects.get_or_create(
            name='Tài liệu FAQ',
            defaults={'description': 'Database chứa các câu hỏi thường gặp và câu trả lời'}
        )
        if created:
            self.stdout.write(f'✓ Tạo database: {db1.name}')
        
        db2, created = QdrantDatabase.objects.get_or_create(
            name='Kiến thức sản phẩm',
            defaults={'description': 'Database chứa thông tin chi tiết về sản phẩm và dịch vụ'}
        )
        if created:
            self.stdout.write(f'✓ Tạo database: {db2.name}')
        
        db3, created = QdrantDatabase.objects.get_or_create(
            name='Hướng dẫn sử dụng',
            defaults={'description': 'Database chứa các hướng dẫn và tutorial'}
        )
        if created:
            self.stdout.write(f'✓ Tạo database: {db3.name}')
        
        # Tạo documents mẫu cho database FAQ
        faq_docs = [
            {
                'qdrant_id': 'faq_001',
                'content': 'Câu hỏi: Làm thế nào để đăng ký tài khoản?\\nTrả lời: Bạn có thể đăng ký tài khoản bằng cách click vào nút "Đăng ký" ở góc trên bên phải của trang web.',
                'link_source': 'https://example.com/faq#dang-ky',
                'file_name': 'FAQ_DangKy.txt'
            },
            {
                'qdrant_id': 'faq_002',
                'content': 'Câu hỏi: Tôi quên mật khẩu, phải làm sao?\\nTrả lời: Bạn có thể reset mật khẩu bằng cách click vào "Quên mật khẩu" ở trang đăng nhập và làm theo hướng dẫn.',
                'link_source': 'https://example.com/faq#quen-mat-khau',
                'file_name': 'FAQ_MatKhau.txt'
            },
            {
                'qdrant_id': 'faq_003',
                'content': 'Câu hỏi: Làm thế nào để liên hệ hỗ trợ?\\nTrả lời: Bạn có thể liên hệ hỗ trợ qua email support@example.com hoặc hotline 1900-xxxx.',
                'link_source': 'https://example.com/contact',
                'file_name': 'FAQ_HoTro.txt'
            }
        ]
        
        for doc_data in faq_docs:
            doc, created = QdrantDocument.objects.get_or_create(
                database=db1,
                qdrant_id=doc_data['qdrant_id'],
                defaults={
                    'content': doc_data['content'],
                    'link_source': doc_data['link_source'],
                    'file_name': doc_data['file_name']
                }
            )
            if created:
                self.stdout.write(f'  ✓ Tạo document: {doc.qdrant_id}')
        
        # Tạo documents mẫu cho database Sản phẩm
        product_docs = [
            {
                'qdrant_id': 'prod_001',
                'content': 'Sản phẩm A là giải pháp quản lý dữ liệu hiệu quả với khả năng xử lý hàng triệu records mỗi giây. Tích hợp AI và machine learning để tối ưu hóa hiệu suất.',
                'link_source': 'https://example.com/products/product-a',
                'file_name': 'SanPham_A.pdf'
            },
            {
                'qdrant_id': 'prod_002',
                'content': 'Dịch vụ tư vấn chuyên nghiệp với đội ngũ chuyên gia hơn 10 năm kinh nghiệm trong lĩnh vực công nghệ thông tin và chuyển đổi số.',
                'link_source': 'https://example.com/services/consulting',
                'file_name': 'DichVu_TuVan.pdf'
            }
        ]
        
        for doc_data in product_docs:
            doc, created = QdrantDocument.objects.get_or_create(
                database=db2,
                qdrant_id=doc_data['qdrant_id'],
                defaults={
                    'content': doc_data['content'],
                    'link_source': doc_data['link_source'],
                    'file_name': doc_data['file_name']
                }
            )
            if created:
                self.stdout.write(f'  ✓ Tạo document: {doc.qdrant_id}')
        
        # Tạo documents mẫu cho database Hướng dẫn
        guide_docs = [
            {
                'qdrant_id': 'guide_001',
                'content': 'Hướng dẫn cài đặt:\\n1. Tải file cài đặt từ trang chủ\\n2. Chạy file setup.exe với quyền Administrator\\n3. Làm theo hướng dẫn trên màn hình\\n4. Khởi động lại máy tính sau khi cài đặt xong.',
                'link_source': 'https://example.com/guides/installation',
                'file_name': 'HD_CaiDat.docx'
            },
            {
                'qdrant_id': 'guide_002',
                'content': 'Hướng dẫn sử dụng cơ bản:\\n- Đăng nhập vào hệ thống\\n- Tạo project mới\\n- Thêm thành viên vào team\\n- Quản lý tasks và deadlines\\n- Xuất báo cáo tiến độ.',
                'link_source': 'https://example.com/guides/basic-usage',
                'file_name': 'HD_SuDung.docx'
            }
        ]
        
        for doc_data in guide_docs:
            doc, created = QdrantDocument.objects.get_or_create(
                database=db3,
                qdrant_id=doc_data['qdrant_id'],
                defaults={
                    'content': doc_data['content'],
                    'link_source': doc_data['link_source'],
                    'file_name': doc_data['file_name']
                }
            )
            if created:
                self.stdout.write(f'  ✓ Tạo document: {doc.qdrant_id}')
        
        self.stdout.write(
            self.style.SUCCESS('✅ Hoàn thành tạo dữ liệu mẫu!')
        )
        
        # Thống kê
        total_dbs = QdrantDatabase.objects.count()
        total_docs = QdrantDocument.objects.count()
        
        self.stdout.write(f'📊 Thống kê:')
        self.stdout.write(f'   - Databases: {total_dbs}')
        self.stdout.write(f'   - Documents: {total_docs}')
