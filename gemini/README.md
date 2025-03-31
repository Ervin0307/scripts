# How to Get Your Gemini API Key

Follow this step-by-step guide to obtain your Gemini API key and start using Google's Gemini AI capabilities.

## Steps to Create a Gemini API Key

### Step 1: Log in to Your Google Account
- Visit [Google AI Studio](https://aistudio.google.com/).
- Log in using your Google account credentials. If you don’t have an account, create one and log in.

### Step 2: Access the Gemini API Section
- Navigate to the "Gemini API" tab or click on the “Get API Key” button on the homepage.
- You can also directly search for "Gemini API key" in Google and follow the relevant link.

### Step 3: Accept Terms and Conditions
- Review the Google APIs Terms of Service and Gemini API Additional Terms of Service.
- Check the required boxes to accept the terms. You may also opt-in for email updates, though this is optional.

### Step 4: Generate Your API Key
- Click on the “Create API Key” button.
- Choose whether to create the key in an existing project or a new project.
- Once selected, your API key will be auto-generated.

### Step 5: Save Your API Key
- **Important**: Copy your API key immediately after it is generated.
- Store it securely, as you won’t be able to view the full key again later.

## Setting Up Your API Key

For secure usage, save your API key as an environment variable:

1. Create a `.env` file in your project directory.
2. Add the following line to the `.env` file:

    ```
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

3. Save the file.

Alternatively, you could also export the variable:
   ```bash
   export GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Security Considerations

To ensure your API key is secure:
- Treat it as confidential information.
- Avoid sharing it publicly or committing it to version control systems like GitHub.
- If compromised, revoke and regenerate a new key immediately.

## Additional Notes

- **Free Tier**: Gemini offers a free plan with rate limits, making it ideal for testing and small-scale applications.
- **Billing**: For higher usage, consider setting up billing for extended features and higher rate limits.

With these steps completed, you're ready to integrate Gemini AI into your applications!
