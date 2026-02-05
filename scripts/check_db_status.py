import sys
import os
from pathlib import Path

# إضافة مجلد المشروع للمسارات
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# استيراد الإعدادات
from src import config
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def check_database():
    print("="*50)
    print("🕵️  بدء فحص قاعدة البيانات (ChromaDB Diagnostic)")
    print("="*50)

    # 1. طباعة المسارات الحالية
    print(f"📂 مسار المشروع (BASE_DIR): {BASE_DIR}")
    print(f"📂 مسار قاعدة البيانات المستهدف: {config.CHROMA_PATH}")
    print(f"🏷️  اسم المجموعة في الكود (Collection Name): {config.COLLECTION_NAME}")

    # 2. التحقق من وجود الملفات
    if not os.path.exists(config.CHROMA_PATH):
        print("❌ خطأ كارثي: المجلد غير موجود أصلاً! تأكد من نقل مجلد chroma_db يدوياً.")
        return

    # 3. محاولة قراءة المجموعات الموجودة فعلياً باستخدام Chroma Client المباشر
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        collections = client.list_collections()
        actual_names = [c.name for c in collections]
        
        print(f"\n📦 المجموعات الموجودة فعلياً داخل قاعدة البيانات:")
        if not actual_names:
            print("   ⚠️  تنبيه: قاعدة البيانات فارغة تماماً (لا توجد مجموعات).")
        else:
            for name in actual_names:
                count = client.get_collection(name).count()
                print(f"   - الاسم: '{name}' | عدد الوثائق: {count}")

        # 4. مقارنة الأسماء
        if config.COLLECTION_NAME not in actual_names:
            print(f"\n❌ مشكلة تطابق: الكود يبحث عن '{config.COLLECTION_NAME}' لكنها غير موجودة!")
            if actual_names:
                print(f"💡 الحل المقترح: عدل COLLECTION_NAME في ملف .env أو config.py ليصبح '{actual_names[0]}'")
        else:
            print(f"\n✅ تطابق الاسم صحيح: المجموعة '{config.COLLECTION_NAME}' موجودة.")

    except Exception as e:
        print(f"❌ حدث خطأ أثناء فحص العميل المباشر: {e}")

    print("="*50)

if __name__ == "__main__":
    check_database()