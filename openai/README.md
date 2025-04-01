# How to Get an OpenAI API Key

Follow these steps to generate and retrieve your OpenAI API key:

## Step 1: Navigate to the OpenAI Platform
- Go to [https://platform.openai.com](https://platform.openai.com).
- If you don't have an account, click **"Sign up"** to create one.
- If you already have an account, click **"Log in"**.

## Step 2: Create or Log into Your Account
- You can sign up/log in using:
  - Email and password
  - Google account
  - Microsoft account
  - Apple account
- If registering with email, verify your account by clicking the verification link sent to your inbox.

## Step 3: Access the API Keys Section
- Once logged in, you'll see the OpenAI dashboard.
- On the left sidebar, click on **"API keys"**.
- If you're a new user, you may need to verify your phone number first by clicking **"Start verification"** and entering the code sent to your phone.

## Step 4: Create a New API Key
- Click the green **"Create new secret key"** button.
- Optionally, give your key a name (e.g., "My Project Key").
- For permissions, you can select:
  - **All** for no restrictions.
  - **Restricted** to control specific permissions.
  - **Read Only** for read-only activities.
- Click **"Create secret key"**.

## Step 5: Copy and Store Your API Key
- **Important**: Copy your API key immediately when it appears.
- This is the only time you'll be able to see the full key.
- Store it securely in a safe place (e.g., in an environment variable or a secure password manager).
- Click **"Done"** after copying the key.

## Step 6: Set Up Payment (Required for API Usage)
- Navigate to **Settings > Billing > Payment methods**.
- Add a payment method as OpenAI charges for API usage.
- For casual users, $20 worth of tokens is typically enough for several months of testing.

---
## Setting Up Your API Key
OpenAI's API is **NOT** free to use for developers. OpenAI operates on a Pay-As-You-Go pricing model where users pay for the API requests they make


For enhanced security and ease of use, it's recommended to set up your API key as an environment variable:

1. Create a `.env` file in the root directory of your project.
2. Add the following line to your `.env` file:

    ```bash
    OPENAI_API_KEY=your_openai_api_key_here
    ```

3. Save the file.


Alternatively, you could also export the variable:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

---

### Notes:
1. If you lose your API key, you cannot retrieve it. You will need to revoke the old key and create a new one.
2. Avoid sharing or exposing your API key in public repositories or codebases.

You are now ready to use OpenAI's API with your generated key!
