# How to Get the Mistral AI API Key (Free Tier)

Mistral AI offers a free tier for developers to experiment and innovate with their AI models. Follow the steps below to obtain your API key and start using their services.

---

## **Step 1: Create or Log in to Your Mistral AI Account**
- Go to [Mistral AI's Registration Page](https://auth.mistral.ai/ui/registration).
- If you already have an account, log in. Otherwise, create a new account by following the on-screen instructions.

---

## **Step 2: Set Up Your Workspace**
- After logging in, set up your workspace:
  - Choose a name for your workspace.
  - Specify whether it’s for individual or team use.
  - Accept Mistral AI’s terms and conditions.

---

## **Step 3: Navigate to the API Keys Section**
- Once your workspace is set up, go to the [API Keys Page](https://console.mistral.ai/api-keys/).
- You can also access this page from the "Overview" section in your account dashboard by selecting "API Keys."

---

## **Step 4: Create a New API Key**
- Click on **"Create new key"**.
- In the pop-up window:
  - Assign a recognizable name to your key (e.g., "Test Key").
  - Optionally, set an expiration date for security purposes.
- Click **"Create key"**.

---

## **Step 5: Copy and Save Your API Key**
- Once created, your API key will be displayed. **Important:** This key will only be shown once.
- Copy the key and save it securely (e.g., in a password manager).
- The key may take a few minutes to activate.

---
## Setting Up Your API Key

For enhanced security and ease of use, it's recommended to set up your API key as an environment variable:

1. Create a `.env` file in the root directory of your project.
2. Add the following line to your `.env` file:

    ```bash
    MISTRAL_API_KEY=your_mistral_api_key_here
    ```

3. Save the file.


Alternatively, you could also export the variable:
   ```bash
   export MISTRAL_API_KEY=your_mistral_api_key_here
   ```

---

## **Details About the Free Tier**
Mistral AI introduced a free tier aimed at developers who want to test and prototype applications. Here are the key features of the free tier:

### **Key Features**
1. **Free API Access**:
   - Use Mistral's models like Mixtral 8x7B for testing and experimentation.
   - Available via their platform "la Plateforme."

2. **Rate Limits**:
   - Limited requests per second or tokens per month (specific limits depend on your usage).

3. **Supported Models**:
   - Includes open-source models like Mistral-Tiny, Mistral-Small, and Mistral-Medium.

4. **Multimodal Capabilities**:
   - Test text-based tasks and image processing features (via le Chat).

5. **No Payment Required**:
   - No credit card is needed for free-tier access.

---

## **Additional Notes**
- For production-level use or higher rate limits, you may need to upgrade to a paid plan.
- The free tier is ideal for developers experimenting with generative AI solutions without incurring costs.

For further assistance, refer to [Mistral AI's Documentation](https://docs.mistral.ai/getting-started/quickstart/) or contact their support team.

---
