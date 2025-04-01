# How to Get Your Together AI API Key

Follow these steps to obtain your API key for Together AI:

## Step 1: Create an Account
1. Visit the Together AI website: [https://api.together.ai](https://api.together.ai).
2. Sign up for an account by providing your email and other required details.
3. Upon account creation, you will receive $1 in free credits to get started.

## Step 2: Access the API Key
1. Log in to your Together AI account.
2. Click on your avatar (profile icon) located in the bottom-left corner of the dashboard.
3. Select **Settings** from the menu.
4. Navigate to the **API Keys** section.
5. Your API key will be displayed here. If you are accessing it for the first time, it may also appear on the initial welcome screen after signing up.

## Step 3: Save Your API Key
1. Copy the API key and store it securely.
2. You can use this key in your projects by passing it as an environment variable (`TOGETHER_API_KEY`) or directly in your code.

---
## Setting Up Your API Key

For enhanced security and ease of use, it's recommended to set up your API key as an environment variable:

1. Create a `.env` file in the root directory of your project.
2. Add the following line to your `.env` file:

    ```bash
    TOGETHER_API_KEY=your_together_api_key_here
    ```

3. Save the file.


Alternatively, you could also export the variable:
   ```bash
   export TOGETHER_API_KEY=your_together_api_key_here
   ```

---


---

## Notes
- Free-tier users have a limit of **60 requests per minute**.
- If you need higher limits or additional features, you can upgrade to a paid account.
- For more details, visit Together AI's support page or contact their support team.

