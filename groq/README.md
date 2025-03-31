# How to Get Your Groq API Key

Getting your Groq API key is a straightforward process. This guide will walk you through the necessary steps to obtain and securely store your API key.

## Creating an Account

1. Visit the Groq Cloud website at [console.groq.com](https://console.groq.com/login).
2. Create a new account if you don't have one already.
3. Complete the verification process if required.

## Obtaining Your API Key

### Step 1: Log in to Your Account
- Sign in to your Groq Cloud console using your credentials.

### Step 2: Navigate to API Keys Section
- Once logged in, look for the "API Keys" section in the console dashboard.

### Step 3: Generate a New API Key
- Click on the "Create API Key" button.
- In the pop-up window, enter a descriptive name for your key (e.g., "Project Name" or "Personal Use").
- Click "Submit" or "Create" to generate your API key.

### Step 4: Save Your API Key
- **Important**: Copy your API key immediately after it's generated.
- Store it in a secure location as you won't be able to view the full key again after leaving this page.

## Setting Up Your API Key

For enhanced security and ease of use, it's recommended to set up your API key as an environment variable:

1. Create a `.env` file in the root directory of your project.
2. Add the following line to your `.env` file:

    ```bash
    GROQ_API_KEY=your_groq_api_key_here
    ```

3. Save the file.


Alternatively, you could also export the variable:
   ```bash
   export GROQ_API_KEY=your_groq_api_key_here
   ```

## Security Considerations

Remember that your API key functions like a password:
- Treat it as confidential information.
- Never share it publicly or commit it directly in your code.
- Only you should have access to it.
- If compromised, others may be able to use Groq Cloud services using your account.

With your API key set up, you're now ready to start using Groq's AI models in your projects.
