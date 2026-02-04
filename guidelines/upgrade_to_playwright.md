# 🚀 دليل ترقية Agent للبحث الديناميكي الكامل

**الهدف**: ترقية `latest_agent.py` لاستخدام Playwright للبحث الفعلي في موقع SFDA

---

## 📋 الخطوات

### الخطوة 1️⃣: تثبيت Playwright

```bash
# تثبيت المكتبة
pip install playwright

# تثبيت المتصفحات (Chrome, Firefox, Safari)
playwright install

# أو فقط Chromium (أخف):
playwright install chromium
```

**ماذا يحدث؟**
- يتم تحميل متصفح Chromium (~300 MB)
- يُحفظ في مجلد خاص بـ Playwright
- يمكن استخدامه برمجياً

---

### الخطوة 2️⃣: فهم بنية موقع SFDA

قبل كتابة الكود، يجب فحص الموقع:

#### أ) افتح الموقع في Chrome
```
https://www.sfda.gov.sa/ar/drugs-list
```

#### ب) افتح Developer Tools (F12)

#### ج) افحص عناصر نموذج البحث:

**مثال على ما ستجده:**

```html
<!-- مثال - يجب التحقق من الموقع الفعلي -->
<div class="search-section">
  <input
    id="registration-number"
    name="regNumber"
    placeholder="رقم التسجيل"
    class="form-control"
  />

  <button class="btn-search" onclick="searchDrugs()">
    بحث
  </button>
</div>

<!-- النتائج -->
<table id="results-table" class="drugs-table">
  <thead>
    <tr>
      <th>الاسم التجاري</th>
      <th>الاسم العلمي</th>
      <th>رقم التسجيل</th>
      <th>الشركة الصانعة</th>
      <th>الوكيل</th>
    </tr>
  </thead>
  <tbody>
    <!-- النتائج هنا -->
  </tbody>
</table>
```

#### د) سجل المعلومات المهمة:

- ✅ **selector حقل رقم التسجيل**: `#registration-number` أو `input[name="regNumber"]`
- ✅ **selector زر البحث**: `.btn-search`
- ✅ **selector جدول النتائج**: `#results-table`
- ✅ **بنية الجدول**: أسماء الأعمدة ومواقعها

---

### الخطوة 3️⃣: تعديل `latest_agent.py`

#### أ) إضافة import للـ Playwright:

```python
# في أعلى الملف، أضف:
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
```

#### ب) تعديل دالة `search_sfda_drug`:

استبدل الدالة الحالية (السطر 25-85) بهذه:

```python
@tool
def search_sfda_drug(registration_number: str) -> str:
    """
    البحث عن دواء في موقع الهيئة العامة للغذاء والدواء السعودية.

    Args:
        registration_number: رقم تسجيل الدواء في الهيئة

    Returns:
        معلومات الدواء إذا كان مسجلاً، أو رسالة خطأ إذا لم يتم العثور عليه
    """
    try:
        print(f"🔍 جاري البحث عن دواء برقم التسجيل: {registration_number}")

        with sync_playwright() as p:
            # فتح المتصفح (headless=True للعمل في الخلفية)
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # الذهاب لموقع SFDA
            page.goto("https://www.sfda.gov.sa/ar/drugs-list", timeout=30000)

            # انتظار تحميل الصفحة بالكامل
            page.wait_for_load_state("networkidle")

            # ⚠️ هنا يجب تعديل selectors حسب الموقع الفعلي
            # هذه أمثلة - يجب فحص الموقع

            # ملء حقل رقم التسجيل
            registration_input = page.wait_for_selector(
                'input[name="registration_number"]',  # عدّل هذا
                timeout=10000
            )
            registration_input.fill(registration_number)

            # الضغط على زر البحث
            search_button = page.query_selector('button.search-btn')  # عدّل هذا
            search_button.click()

            # انتظار ظهور النتائج
            page.wait_for_selector('table.results', timeout=15000)  # عدّل هذا

            # قراءة النتائج
            rows = page.query_selector_all('table.results tbody tr')

            if len(rows) == 0:
                browser.close()
                return """
❌ لم نجد دواء مسجل بهذا الرقم.

💡 تأكد من:
- صحة رقم التسجيل
- كتابة الرقم بشكل صحيح
- أن الدواء مسجل في الفئة الصحيحة (بشري/بيطري/عشبي)
                """

            # استخراج بيانات أول نتيجة
            first_row = rows[0]
            cells = first_row.query_selector_all('td')

            # تنسيق المعلومات
            result = f"""
✅ تم العثور على الدواء!

📋 معلومات الدواء:
{'='*60}
🏷️  الاسم التجاري: {cells[0].inner_text().strip()}
💊 الاسم العلمي: {cells[1].inner_text().strip()}
🔢 رقم التسجيل: {cells[2].inner_text().strip()}
🏭 الشركة الصانعة: {cells[3].inner_text().strip()}
🏢 الوكيل: {cells[4].inner_text().strip()}
{'='*60}

🌐 المصدر: الهيئة العامة للغذاء والدواء السعودية
            """

            browser.close()
            return result

    except PlaywrightTimeout:
        return """
⏰ انتهت مهلة الاتصال بموقع الهيئة.

الأسباب المحتملة:
- الموقع بطيء في الاستجابة
- مشكلة في الاتصال بالإنترنت
- الموقع قيد الصيانة

💡 حاول مرة أخرى بعد قليل.
        """

    except Exception as e:
        return f"""
❌ خطأ أثناء البحث: {str(e)}

💡 قد يكون السبب:
- تغيير في بنية موقع SFDA
- يجب تحديث selectors في الكود
- مشكلة في تثبيت Playwright

🔧 للتحقق: راجع ملف upgrade_to_playwright.md
        """
```

---

### الخطوة 4️⃣: اختبار الكود

```bash
python latest_agent.py
```

إذا ظهر خطأ، اتبع هذه الخطوات:

#### أ) تشغيل بوضع "مرئي" للتصحيح:

غيّر:
```python
browser = p.chromium.launch(headless=True)
```

إلى:
```python
browser = p.chromium.launch(headless=False, slow_mo=1000)
# slow_mo=1000 يُبطئ الحركات لتراها
```

#### ب) أضف screenshots للتصحيح:

```python
# بعد كل خطوة:
page.screenshot(path='step1_loaded.png')
registration_input.fill(registration_number)
page.screenshot(path='step2_filled.png')
search_button.click()
page.screenshot(path='step3_clicked.png')
```

#### ج) طباعة HTML للفحص:

```python
# إذا لم تجد عنصر:
print(page.content())  # طباعة كل HTML
```

---

### الخطوة 5️⃣: تعديل selectors حسب الموقع الفعلي

بعد فحص الموقع، عدّل:

```python
# بدلاً من:
registration_input = page.wait_for_selector('input[name="registration_number"]')

# استخدم selector الصحيح، مثل:
registration_input = page.wait_for_selector('#regNum')  # مثال
# أو
registration_input = page.wait_for_selector('input.registration-field')  # مثال
```

---

## 🎯 كيف تجد selector الصحيح؟

### طريقة 1: من Developer Tools

1. افتح الموقع
2. F12 لفتح Developer Tools
3. اضغط على أيقونة المؤشر (Inspect)
4. اضغط على العنصر (مثلاً حقل رقم التسجيل)
5. في الأسفل سترى HTML

```html
<input id="drugRegNum" name="registration" class="form-control">
```

**selectors الممكنة:**
- `#drugRegNum` (بالـ id - الأفضل!)
- `input[name="registration"]` (بالـ name)
- `.form-control` (بالـ class - قد يكون غير دقيق)

### طريقة 2: باستخدام Console

في Developer Tools، اذهب لـ Console واكتب:

```javascript
// اختبر selector:
document.querySelector('#drugRegNum')

// إذا رجع null، جرّب غيره:
document.querySelector('input[name="registration"]')
```

---

## 🔍 مثال كامل على التعديل

### قبل (الكود الحالي):

```python
# يستخدم httpx - محدود
with httpx.Client() as client:
    response = client.get(url)
    # لا يمكن التفاعل مع JavaScript
```

### بعد (مع Playwright):

```python
# يستخدم Playwright - قوي
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url)

    # يمكن ملء الحقول
    page.fill('#regNum', registration_number)

    # يمكن الضغط على الأزرار
    page.click('button.search')

    # يمكن انتظار النتائج
    page.wait_for_selector('table.results')

    # يمكن قراءة البيانات الديناميكية
    results = page.query_selector_all('table tr')
```

---

## 📊 مقارنة شاملة

| الميزة | HTTPX (حالي) | Playwright (مطلوب) |
|--------|--------------|-------------------|
| تشغيل JavaScript | ❌ لا | ✅ نعم |
| ملء النماذج | ❌ لا | ✅ نعم |
| الضغط على الأزرار | ❌ لا | ✅ نعم |
| انتظار النتائج الديناميكية | ❌ لا | ✅ نعم |
| قراءة محتوى AJAX | ❌ لا | ✅ نعم |
| أخذ Screenshots | ❌ لا | ✅ نعم |
| سرعة | ⚡ سريع جداً | 🐌 أبطأ قليلاً |
| استهلاك الموارد | 💚 قليل | 💛 متوسط |
| سهولة الاستخدام | ✅ سهل | ✅ سهل نسبياً |

---

## 🎓 ملخص

### لماذا نحتاج Playwright؟

```
موقع SFDA:
   HTML ← JavaScript يضيف نموذج البحث
                    ↓
            عند الضغط على "بحث"
                    ↓
            AJAX يطلب البيانات
                    ↓
            JavaScript يعرض النتائج

HTTPX: يرى فقط HTML الأساسي ❌
Playwright: يشغل JavaScript ويرى كل شيء ✅
```

### الخطوات التالية:

1. ✅ ثبّت Playwright: `pip install playwright && playwright install`
2. ✅ افحص موقع SFDA وسجل selectors
3. ✅ عدّل `search_sfda_drug` في `latest_agent.py`
4. ✅ اختبر بوضع `headless=False` أولاً
5. ✅ عدّل selectors حسب الحاجة
6. ✅ غيّر إلى `headless=True` للإنتاج

---

## 💡 نصائح إضافية

### 1. التعامل مع التبويبات (إذا كان الموقع فيه فئات):

```python
# اضغط على تبويب "الأدوية البشرية"
page.click('a[href="#human-drugs"]')
page.wait_for_timeout(1000)  # انتظر ثانية

# ثم ابحث
page.fill('#regNum', registration_number)
```

### 2. التعامل مع Dropdowns:

```python
# إذا كان هناك قائمة اختيار
page.select_option('select#drug-category', 'human')
```

### 3. التعامل مع Pop-ups:

```python
# إذا كان هناك نافذة منبثقة
page.on('dialog', lambda dialog: dialog.accept())
```

### 4. الانتظار الذكي:

```python
# بدلاً من wait_for_timeout (ثابت):
page.wait_for_timeout(5000)  # ❌ ينتظر 5 ث حتى لو انتهى قبل

# استخدم:
page.wait_for_selector('table.results')  # ✅ ينتظر حتى يظهر فقط
```

---

**جاهز للتطبيق؟ ابدأ من الخطوة 1! 🚀**
