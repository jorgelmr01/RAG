# Getting Started - Simple Guide

## For People New to This

This is a simple document assistant that lets you upload files and ask questions about them. No technical knowledge needed!

---

## Step 1: Install Python (if you don't have it)

**Windows:**
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or newer
3. **IMPORTANT:** When installing, check the box that says "Add Python to PATH"
4. Click "Install Now"

**Mac:**
1. Go to https://www.python.org/downloads/mac-osx/
2. Download Python 3.11 or newer
3. Run the installer and follow the instructions

**How to check if you have Python:**
- Windows: Open Command Prompt, type `python --version` and press Enter
- Mac: Open Terminal, type `python3 --version` and press Enter
- If you see a version number (like "Python 3.11.5"), you're good!

---

## Step 2: Get Your OpenAI API Key

1. Go to https://platform.openai.com/signup
2. Create an account (or sign in if you have one)
3. Go to https://platform.openai.com/account/api-keys
4. Click "Create new secret key"
5. Copy the key (it starts with "sk-")
6. **Save it somewhere safe** - you'll need it in Step 4

---

## Step 3: Download and Unzip This Project

1. Download the project folder as a ZIP file
2. Right-click the ZIP file and select "Extract All" (Windows) or double-click it (Mac)
3. Open the extracted folder

---

## Step 4: Start the App

**Windows:**
- Double-click `start_app.bat`
- Wait for it to open in your browser (this may take 1-2 minutes the first time)

**Mac:**
- Right-click `start_app.command` and select "Open"
- If you get a security warning, go to System Settings → Privacy & Security → click "Open Anyway"
- Wait for it to open in your browser

---

## Step 5: Enter Your API Key

1. When the app opens in your browser, click on "Configuration" at the top
2. Paste your API key (the one that starts with "sk-") into the "OpenAI API Key" box
3. Click "Set API Key"
4. You should see a green checkmark ✅

---

## Step 6: Upload Documents and Ask Questions

1. Click "Projects" at the top
2. Type a project name (like "my-documents") and click "Create & Switch"
3. Click "Upload documents" and select your files (PDF, Word, text files work best)
4. Click "Process Documents" and wait for it to finish
5. Type your question in the chat box and press Enter
6. Get your answer! 🎉

---

## Common Questions

**Q: The app won't start. What do I do?**
- Make sure Python is installed (see Step 1)
- Make sure you checked "Add Python to PATH" during installation
- Try restarting your computer

**Q: It says "Invalid API key". What's wrong?**
- Make sure you copied the entire key (it should be about 51 characters long)
- Make sure it starts with "sk-"
- Try copying it again from https://platform.openai.com/account/api-keys

**Q: How much does this cost?**
- The app uses OpenAI's API, which charges based on usage
- Small documents cost just a few cents
- Check your usage at https://platform.openai.com/usage

**Q: Can I share this with friends?**
- Yes! Just zip the folder (but don't include the `.venv` folder if it exists)
- Your friends need to follow these same steps

---

## Need Help?

If something doesn't work:
1. Check that Python is installed correctly
2. Make sure your API key is valid
3. Try restarting the app
4. Check the error message - it usually tells you what's wrong

