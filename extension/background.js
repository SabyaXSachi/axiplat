// Background worker keeps Instagram blocked until the local gate grants access
const GATE_URL = 'http://localhost:5000';
const INSTAGRAM_DOMAINS = ['www.instagram.com', 'instagram.com'];

async function isServerRunning() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000); // bail out quickly if server is dead
    
    const response = await fetch(`${GATE_URL}/status`, { 
      method: 'GET',
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    return false;
  }
}

async function isVerified() {
  const serverRunning = await isServerRunning();
  if (!serverRunning) {
    return false;
  }

  const result = await chrome.storage.local.get(['age_gate_token']);
  const token = result.age_gate_token;
  
  if (!token) {
    return false;
  }

  try {
    const response = await fetch(`${GATE_URL}/api/check_verification?token=${encodeURIComponent(token)}`);
    const data = await response.json();
    return data.verified === true;
  } catch (error) {
    console.error('Verification check failed:', error);
    return false;
  }
}

async function checkOnceOnStart() {
  const result = await chrome.storage.local.get(['age_gate_token']);
  const verified = await isVerified();
  
  if (!verified) {
    chrome.tabs.query({ url: ['https://www.instagram.com/*', 'https://instagram.com/*'] }, (tabs) => {
      if (tabs.length > 0) {
        chrome.tabs.update(tabs[0].id, { url: GATE_URL });
      }
    });
  }
}

checkOnceOnStart();
