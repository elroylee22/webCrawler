import os
import json
import time
import asyncio
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
from playwright.async_api import async_playwright, Page
import signal

# === Load Environment ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PRISMA_URL = os.getenv("PRISMA_URL")
START_ID = 0
MAX_CONCURRENT = 10

# === Set OpenAI key ===
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# === PostgreSQL Connection ===
conn = psycopg2.connect(PRISMA_URL)
cur = conn.cursor()

# === Graceful Stop ===
stop_requested = False

def handle_sigint(signum, frame):
    global stop_requested
    stop_requested = True
    print("\n🛑 Stop requested. Finishing current batch...")

signal.signal(signal.SIGINT, handle_sigint)

# === DB and Helper Functions ===
def fetch_companies(limit=10, start_from=0):
    cur.execute("""
        SELECT id, name, website 
        FROM companies 
        WHERE website IS NOT NULL AND product_name IS NULL AND id >= %s
        ORDER BY id
        LIMIT %s;
    """, (start_from, limit))
    return cur.fetchall()

def normalize_field(value):
    if isinstance(value, list):
        value = ", ".join(map(str, value))
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False)
    elif value is None:
        return ""
    return str(value).encode().decode('unicode_escape').strip()

def save_to_db(company_id, gpt_data):
    try:
        cur.execute("""
            UPDATE companies
            SET 
                product_name = %s,
                product_function = %s,
                product_location = %s,
                product_qual = %s,
                updated_at = NOW()
            WHERE id = %s;
        """, (
            normalize_field(gpt_data.get("product_name")),
            normalize_field(gpt_data.get("product_function")),
            normalize_field(gpt_data.get("product_location")),
            normalize_field(gpt_data.get("product_qual")),
            company_id
        ))
        conn.commit()
        print(f"✅ Saved to DB (Company ID {company_id})")
    except Exception as e:
        print(f"❌ DB Save Error for Company ID {company_id}: {e}")

# === AI Extraction Functions ===
async def translate_to_english_if_needed(raw_text):
    prompt = f"""
You are a translation engine. If the text is NOT English, translate it fully to English.
If it is already English, return it exactly as is.

Text:
{raw_text[:1500]}
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        translated = response.choices[0].message.content.strip()
        return translated
    except Exception as e:
        print(f"⚠️ Translation Error: {e}")
        return raw_text

def extract_with_gpt(english_text):
    trimmed_text = english_text[:6000]  # Still trimmed for cost optimization
    prompt = f"""
You are analyzing a company's website content.

Please extract the following:
1. "product_name": List ALL products/services mentioned. Be very detailed. E.g., "Jeans, Shirts, Jackets", not just "Clothes".
2. "product_function": For each product/service, describe clearly what it is used for. Be as detailed as possible.
3. "product_location": Where the company operates or provides services (e.g., specific cities, regions, countries).
4. "product_qual": Any certifications, awards, industry standards, or notable achievements mentioned.

Return STRICTLY valid JSON in this format:

{{
  "product_name": "...",
  "product_function": "...",
  "product_location": "...",
  "product_qual": "..."
}}

⚠️ Do not explain anything. Do not add notes. Only output JSON. No surrounding text.

Here is the website text:
\"\"\"
{trimmed_text}
\"\"\"
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content
        try:
            return json.loads(result)
        except json.JSONDecodeError as json_err:
            print(f"❌ GPT JSON parsing error: {json_err}")
            print(f"🔎 GPT Raw output:\n{result}")
            return None
    except Exception as e:
        print(f"❌ GPT API Error: {e}")
        return None

# === Scraping + Processing ===
async def process_company(company, playwright):
    company_id, name, website = company
    print(f"\n🔎 {name} ({website})")

    if not website.startswith("http"):
        print(f"⚠️ Skipping invalid URL: {website}")
        return

    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context()

    try:
        page: Page = await context.new_page()
        try:
            await page.goto(website, timeout=25000)
            await page.wait_for_timeout(1500)
        except Exception as e:
            print(f"❌ Cannot open {website} → {e}")
            cur.execute("""
                UPDATE companies
                SET product_name = %s, product_function = '', product_location = '', product_qual = '', updated_at = NOW()
                WHERE id = %s;
            """, ("[unreachable website]", company_id))
            conn.commit()
            print(f"☠️ Marked as unreachable (Company ID {company_id})")
            return

        if page.is_closed():
            print(f"⚠️ Page already closed for {name}. Skipping.")
            return

        try:
            content = await page.inner_text("body", timeout=8000)
        except Exception as e:
            print(f"⚠️ Content unreadable for {name}. Marking as unreadable.")
            cur.execute("""
                UPDATE companies
                SET product_name = %s, product_function = '', product_location = '', product_qual = '', updated_at = NOW()
                WHERE id = %s;
            """, ("[unreadable content]", company_id))
            conn.commit()
            print(f"☠️ Marked as unreadable (Company ID {company_id})")
            return

        translated_content = await translate_to_english_if_needed(content)
        gpt_result = extract_with_gpt(translated_content)

        if gpt_result:
            print("🧠 Extracted:", json.dumps(gpt_result, indent=2, ensure_ascii=False))
            save_to_db(company_id, gpt_result)
        else:
            print(f"⚠️ GPT failed for {name}. Marking as GPT fail.")
            cur.execute("""
                UPDATE companies
                SET product_name = %s, product_function = '', product_location = '', product_qual = '', updated_at = NOW()
                WHERE id = %s;
            """, ("[gpt fail]", company_id))
            conn.commit()
            print(f"☠️ Marked as GPT fail (Company ID {company_id})")

    except Exception as e:
        print(f"❌ Fatal error for {name}: {e}")
    finally:
        try:
            await context.close()
        except:
            pass
        try:
            await browser.close()
        except:
            pass

# === Main Execution ===
async def main():
    async with async_playwright() as playwright:
        while True:
            if stop_requested:
                print("⏹️ Stop flag active. No more batches will be processed.")
                break

            companies = fetch_companies(limit=MAX_CONCURRENT, start_from=START_ID)
            if not companies:
                print("✅ All companies processed!")
                break

            tasks = [process_company(company, playwright) for company in companies]

            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                print(f"⚠️ Error while gathering tasks: {e}")

        print("👋 Finished all running tasks. Exiting.")

if __name__ == "__main__":
    asyncio.run(main())
