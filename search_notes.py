import os

# User se keyword input lo
search_query = input("🔍 Search Notes: ").lower()

# Sab folders check karo
folders = ["math", "physics", "history", "general"]

found = False
for folder in folders:
    file_path = os.path.join(folder, "notes.txt")
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().lower()
            if search_query in content:
                print(f"✅ Found in: {file_path}")
                print("📝 Note Content:\n", content)
                found = True
                break

if not found:
    print("❌ No matching notes found.")
