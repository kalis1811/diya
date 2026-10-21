const $ = (selector) => document.querySelector(selector);
const conversationBox = $('#messages');
const welcome = $('#welcome');
const questionBox = $('#question');
const form = $('#chat-form');
const fileInput = $('#file-input');
const STORE_KEY = 'diya.local.conversation';
let health = null;
let activeConversation = null;
let messages = [];
let pickedFile = null;
let oldAudioUrl = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

function readableMarkdown(text) {
  const lines = escapeHtml(text).split(/\r?\n/);
  let html = '', listOpen = false;
  const inline = (line) => line
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>');
  for (const raw of lines) {
    const line = raw.trim();
    if (/^[-*]\s+/.test(line)) {
      if (!listOpen) { html += '<ul>'; listOpen = true; }
      html += `<li>${inline(line.replace(/^[-*]\s+/, ''))}</li>`;
      continue;
    }
    if (listOpen) { html += '</ul>'; listOpen = false; }
    if (!line) { html += '<div class="text-gap"></div>'; continue; }
    if (/^#{1,3}\s/.test(line)) html += `<h3>${inline(line.replace(/^#{1,3}\s+/, ''))}</h3>`;
    else html += `<p>${inline(line)}</p>`;
  }
  if (listOpen) html += '</ul>';
  return html;
}

function storeLocal() {
  if (!health?.mongo_ready) {
    try { localStorage.setItem(STORE_KEY, JSON.stringify({id: activeConversation, messages})); }
    catch { /* Browser storage may be disabled; the current chat still works. */ }
  }
}

function scrollBottom() {
  requestAnimationFrame(() => $('.conversation').scrollTo({top: $('.conversation').scrollHeight, behavior: 'smooth'}));
}

function renderMessage(message, {playVoice = false, audioBase64 = null} = {}) {
  welcome.hidden = true;
  const row = document.createElement('article');
  row.className = `message-row ${message.role}`;
  if (message.role === 'assistant') {
    const avatar = document.createElement('div'); avatar.className = 'message-avatar'; avatar.textContent = 'd';
    const body = document.createElement('div'); body.className = 'message-body';
    const name = document.createElement('div'); name.className = 'message-name'; name.textContent = 'Diya';
    const text = document.createElement('div'); text.className = 'message-text'; text.innerHTML = readableMarkdown(message.content);
    body.append(name, text);
    if (message.sources?.length) {
      const sources = document.createElement('div'); sources.className = 'source-list';
      for (const item of message.sources) { const pill = document.createElement('span'); pill.className = 'source-pill'; pill.textContent = item.file || item; sources.append(pill); }
      body.append(sources);
    }
    row.append(avatar, body);
    if (audioBase64) {
      const audio = document.createElement('audio'); audio.className = 'message-audio'; audio.controls = true;
      const bytes = Uint8Array.from(atob(audioBase64), c => c.charCodeAt(0));
      if (oldAudioUrl) URL.revokeObjectURL(oldAudioUrl);
      oldAudioUrl = URL.createObjectURL(new Blob([bytes], {type: 'audio/wav'})); audio.src = oldAudioUrl;
      body.append(audio);
      if (playVoice) audio.play().catch(() => { audio.focus(); });
    }
  } else {
    const body = document.createElement('div'); body.className = 'message-body';
    const text = document.createElement('div'); text.className = 'message-text'; text.textContent = message.content;
    body.append(text); row.append(body);
  }
  conversationBox.append(row); scrollBottom();
}

function showTyping() {
  const row = document.createElement('div'); row.className = 'message-row assistant'; row.id = 'typing-row';
  row.innerHTML = '<div class="message-avatar">d</div><div class="message-body"><div class="message-name">Diya</div><div class="typing">Thinking</div></div>';
  welcome.hidden = true; conversationBox.append(row); scrollBottom();
}

function clearChat() {
  conversationBox.replaceChildren(); messages = []; welcome.hidden = false;
  questionBox.value = ''; pickedFile = null; updatePickedFile(); storeLocal();
}

function pushMessage(role, content, sources = []) {
  const item = {role, content, sources, created_at: new Date().toISOString()};
  messages.push(item); renderMessage(item); storeLocal(); return item;
}

function setStatus(label, detail, kind = '') {
  $('#connection-label').textContent = label;
  $('#connection-detail').textContent = detail;
  $('#connection-dot').className = `connection-dot ${kind}`;
}

async function loadHealth() {
  try {
    const response = await fetch('/health'); health = await response.json();
    $('#model-name').textContent = health.model || 'Local assistant';
    $('#index-status').textContent = health.indexed_chunks ? `${health.indexed_chunks} knowledge pieces` : 'General help is ready';
    const mongoReady = health.storage?.startsWith('MongoDB connected');
    health.mongo_ready = Boolean(mongoReady);
    if (health.status === 'ok' || health.status === 'needs_ingest') {
      setStatus(mongoReady ? 'Assistant ready' : 'Assistant ready · local storage',
        mongoReady ? 'Chats and memories saved in MongoDB' : 'MongoDB is offline; this browser keeps the current chat',
        mongoReady ? 'ready' : '');
    } else setStatus('Assistant needs attention', health.status, 'error');
    $('#voice-toggle').disabled = health.voice?.tts !== 'ready';
    if ($('#voice-toggle').disabled) $('#voice-toggle').checked = false;
  } catch {
    setStatus('Assistant offline', 'Start Diya, then refresh this page', 'error');
    $('#model-name').textContent = 'Not connected';
  }
}

function displayTime(value) {
  if (!value) return 'No time set';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
}

async function loadReminders() {
  const list = $('#reminder-list'); const status = $('#reminder-status');
  list.replaceChildren(); status.textContent = 'Loading reminders…';
  try {
    const response = await fetch('/tasks'); const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Could not load reminders.');
    const reminders = (data.tasks || []).filter(task => task.remind_at && !['completed', 'failed', 'cancelled'].includes(task.status));
    if (!reminders.length) {
      status.textContent = 'No reminders yet. Add one above and Diya will check it while the app is open.';
      return;
    }
    status.textContent = `${reminders.length} active reminder${reminders.length === 1 ? '' : 's'}.`;
    for (const item of reminders) {
      const row = document.createElement('div'); row.className = 'memory-item';
      const reminderState = item.last_nudged_at ? `Reminded ${displayTime(item.last_nudged_at)}` : item.status;
      const detail = document.createElement('span'); detail.textContent = `${item.objective} · ${displayTime(item.remind_at)} · ${reminderState}`;
      const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'reminder-remove';
      remove.textContent = 'Remove'; remove.setAttribute('aria-label', `Remove reminder: ${item.objective}`);
      remove.addEventListener('click', async () => {
        remove.disabled = true;
        try {
          const result = await fetch(`/tasks/${encodeURIComponent(item.task_id)}`, {method: 'DELETE'});
          if (!result.ok) throw new Error('Could not remove this reminder.');
          await loadReminders();
        } catch (error) { status.textContent = error.message; remove.disabled = false; }
      });
      row.append(detail, remove); list.append(row);
    }
  } catch (error) { status.textContent = error.message; }
}

async function checkCaretakerNudges() {
  if (document.visibilityState !== 'visible') return;
  try {
    const response = await fetch('/caretaker/nudges');
    if (!response.ok) return;
    const data = await response.json();
    for (const nudge of data.nudges || []) pushMessage('assistant', `${nudge.message}\n\n${nudge.reason}`);
    if (data.nudges?.length && $('#reminder-dialog').open) await loadReminders();
  } catch { /* Reminder checks must not interrupt chat when the local service is unavailable. */ }
}

function renderRecent(items) {
  const list = $('#recent-conversations'); list.replaceChildren();
  for (const item of items) {
    const button = document.createElement('button'); button.className = 'recent-chat';
    button.textContent = item.title || 'New conversation'; button.title = item.title || '';
    if (item.id === activeConversation) button.classList.add('selected');
    button.addEventListener('click', () => openConversation(item.id)); list.append(button);
  }
}

async function refreshConversations() {
  if (!health?.mongo_ready) return;
  try { const response = await fetch('/conversations'); const data = await response.json(); renderRecent(data.conversations || []); }
  catch { setStatus('MongoDB connection lost', 'Restart MongoDB to save conversations', 'error'); }
}

async function startConversation() {
  if (health?.mongo_ready) {
    const response = await fetch('/conversations', {method: 'POST'});
    if (!response.ok) throw new Error('Could not start a conversation in MongoDB.');
    const item = await response.json(); activeConversation = item.id;
  } else activeConversation = `local-${Date.now()}`;
  clearChat(); await refreshConversations(); questionBox.focus();
}

async function openConversation(id) {
  if (!health?.mongo_ready) return;
  const response = await fetch(`/conversations/${encodeURIComponent(id)}`);
  if (!response.ok) return;
  const data = await response.json(); activeConversation = data.id; conversationBox.replaceChildren();
  messages = data.messages || []; welcome.hidden = messages.length > 0;
  for (const item of messages) renderMessage(item);
  renderRecent((await (await fetch('/conversations')).json()).conversations || []);
}

function updatePickedFile() {
  $('#attachment-row').hidden = !pickedFile;
  $('#file-chip').textContent = pickedFile ? `${pickedFile.name} · ${Math.max(1, Math.round(pickedFile.size / 1024))} KB` : '';
  $('#index-file').disabled = !pickedFile;
}

async function sendFile(file) {
  pushMessage('user', `Please read this file: ${file.name}`);
  showTyping(); $('#send-button').disabled = true;
  const data = new FormData(); data.append('file', file);
  if (health?.mongo_ready && activeConversation) data.append('conversation_id', activeConversation);
  try {
    const response = await fetch('/read-file', {method: 'POST', body: data});
    const result = await response.json(); $('#typing-row')?.remove();
    if (!response.ok) throw new Error(result.detail || 'Diya could not read this file.');
    const answer = result.summary || 'The file was read, but no summary was returned.';
    const item = {role: 'assistant', content: answer, sources: result.sources || [], created_at: new Date().toISOString()};
    messages.push(item); renderMessage(item); storeLocal();
    if (health?.mongo_ready) await refreshConversations();
  } catch (error) {
    $('#typing-row')?.remove(); const row = document.createElement('div'); row.className = 'error-banner'; row.textContent = error.message; conversationBox.append(row); scrollBottom();
  } finally { $('#send-button').disabled = false; }
}

async function sendQuestion(question) {
  pushMessage('user', question); showTyping(); $('#send-button').disabled = true;
  try {
    const useVoice = $('#voice-toggle').checked;
    const endpoint = useVoice ? '/query/voice' : '/query';
    const payload = {question, deep_think: $('#thinking-toggle').checked};
    if (health?.mongo_ready && activeConversation) payload.conversation_id = activeConversation;
    const response = await fetch(endpoint, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const result = await response.json(); $('#typing-row')?.remove();
    if (!response.ok) throw new Error(result.detail || 'Diya could not answer.');
    const item = {role: 'assistant', content: result.answer, sources: result.sources || [], created_at: new Date().toISOString()};
    messages.push(item); renderMessage(item, {playVoice: Boolean(result.audio_base64), audioBase64: result.audio_base64 || null}); storeLocal();
    if (health?.mongo_ready) await refreshConversations();
  } catch (error) {
    $('#typing-row')?.remove(); const row = document.createElement('div'); row.className = 'error-banner'; row.textContent = error.message; conversationBox.append(row); scrollBottom();
  } finally { $('#send-button').disabled = false; questionBox.focus(); }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault(); const text = questionBox.value.trim();
  if (pickedFile) { const file = pickedFile; pickedFile = null; updatePickedFile(); questionBox.value = ''; await sendFile(file); return; }
  if (!text) return; questionBox.value = ''; questionBox.style.height = 'auto'; await sendQuestion(text);
});
questionBox.addEventListener('input', () => { questionBox.style.height = 'auto'; questionBox.style.height = `${Math.min(questionBox.scrollHeight, 180)}px`; });
questionBox.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); } });
$('#new-chat').addEventListener('click', () => startConversation().catch(error => setStatus(error.message, 'Check MongoDB and try again', 'error')));
$('#attach-button').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { pickedFile = fileInput.files?.[0] || null; updatePickedFile(); fileInput.value = ''; questionBox.focus(); });
$('#remove-file').addEventListener('click', () => { pickedFile = null; updatePickedFile(); });
$('#index-file').addEventListener('click', async () => {
  if (!pickedFile) return;
  const file = pickedFile; const button = $('#index-file'); button.disabled = true; button.textContent = 'Indexing…';
  try {
    const data = new FormData(); data.append('file', file);
    const response = await fetch('/knowledge', {method: 'POST', body: data}); const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Could not add this file to the knowledge base.');
    pickedFile = null; updatePickedFile();
    pushMessage('assistant', `${result.message} Diya can use it in future answers.`);
    await loadHealth();
  } catch (error) {
    const row = document.createElement('div'); row.className = 'error-banner'; row.textContent = error.message; conversationBox.append(row); scrollBottom();
  } finally { button.textContent = 'Add to knowledge'; button.disabled = !pickedFile; }
});
document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => { questionBox.value = button.dataset.prompt; questionBox.focus(); }));
document.querySelector('[data-action="upload"]').addEventListener('click', () => fileInput.click());
document.querySelectorAll('[data-copy-prompt]').forEach(button => button.addEventListener('click', async () => {
  const status = $('#copy-status');
  try {
    await navigator.clipboard.writeText(button.dataset.copyPrompt);
    status.textContent = `${button.closest('.idea-card').querySelector('h3').textContent} prompt copied. Paste it into AppDeploy.`;
    button.querySelector('span').textContent = 'Copied';
    window.setTimeout(() => { button.querySelector('span').textContent = 'Copy AppDeploy prompt'; }, 1800);
  } catch {
    const temporary = document.createElement('textarea');
    temporary.value = button.dataset.copyPrompt;
    temporary.setAttribute('readonly', '');
    temporary.style.position = 'fixed'; temporary.style.opacity = '0';
    document.body.append(temporary); temporary.select();
    const copied = document.execCommand('copy'); temporary.remove();
    status.textContent = copied ? 'Prompt copied. Paste it into AppDeploy.' : 'Could not access the clipboard. Try again from a secure browser page.';
    if (copied) {
      button.querySelector('span').textContent = 'Copied';
      window.setTimeout(() => { button.querySelector('span').textContent = 'Copy AppDeploy prompt'; }, 1800);
    }
  }
}));
const mainPanel = $('.main-panel');
mainPanel.addEventListener('dragover', event => { event.preventDefault(); mainPanel.classList.add('dragging'); });
mainPanel.addEventListener('dragleave', event => { if (!mainPanel.contains(event.relatedTarget)) mainPanel.classList.remove('dragging'); });
mainPanel.addEventListener('drop', event => { event.preventDefault(); mainPanel.classList.remove('dragging'); const file = event.dataTransfer.files?.[0]; if (file) { pickedFile = file; updatePickedFile(); } });

$('#about-button').addEventListener('click', async () => {
  try {
    const data = await (await fetch('/about')).json(); $('#about-description').textContent = data.description;
    const facts = $('#about-facts'); facts.replaceChildren();
    for (const [value, label] of [[data.model, 'Language model'], [health?.indexed_chunks || 0, 'Knowledge pieces'], [data.memory_items, 'Saved memories']]) {
      const item = document.createElement('div'); item.className = 'fact'; item.innerHTML = `<strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span>`; facts.append(item);
    }
    $('#info-dialog').showModal();
  } catch { setStatus('Could not load assistant details', 'Refresh and try again', 'error'); }
});
document.querySelectorAll('[data-close]').forEach(button => button.addEventListener('click', () => $(`#${button.dataset.close}`).close()));
for (const dialog of document.querySelectorAll('dialog')) dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

async function loadMemories() {
  const response = await fetch('/memory'); const data = await response.json(); const list = $('#memory-list'); list.replaceChildren();
  if (!data.memories?.length) { const empty = document.createElement('div'); empty.className = 'memory-empty'; empty.textContent = 'Nothing saved yet. Tell Diya “Remember that…” or add a fact below.'; list.append(empty); return; }
  for (const item of data.memories) {
    const row = document.createElement('div'); row.className = 'memory-item'; const text = document.createElement('span'); text.textContent = item.text;
    const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = 'Forget'; remove.addEventListener('click', async () => { await fetch(`/memory/${encodeURIComponent(item.id)}`, {method:'DELETE'}); await loadMemories(); }); row.append(text, remove); list.append(row);
  }
}
$('#show-memories').addEventListener('click', async () => { try { await loadMemories(); $('#memory-dialog').showModal(); } catch { setStatus('Could not load saved memories', 'Check the local API', 'error'); } });

async function loadKnowledge() {
  const response = await fetch('/knowledge'); const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Could not load the knowledge base.');
  const list = $('#knowledge-list'); list.replaceChildren();
  $('#knowledge-status').textContent = `${data.files.length} saved file${data.files.length === 1 ? '' : 's'} · ${data.indexed_chunks} indexed knowledge pieces`;
  if (!data.files.length) { const empty = document.createElement('div'); empty.className = 'memory-empty'; empty.textContent = 'No documents added yet. Choose a file above to start your local knowledge base.'; list.append(empty); return; }
  for (const file of data.files) {
    const row = document.createElement('div'); row.className = 'memory-item';
    const text = document.createElement('span'); text.textContent = `${file.name} · ${Math.max(1, Math.round(file.size / 1024))} KB`;
    const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = 'Remove';
    remove.addEventListener('click', async () => {
      remove.disabled = true;
      try { const result = await fetch(`/knowledge/${encodeURIComponent(file.name)}`, {method: 'DELETE'}); const body = await result.json(); if (!result.ok) throw new Error(body.detail || 'Could not remove this file.'); await loadKnowledge(); await loadHealth(); }
      catch (error) { remove.disabled = false; $('#knowledge-status').textContent = error.message; }
    });
    row.append(text, remove); list.append(row);
  }
}
$('#show-knowledge').addEventListener('click', async () => {
  $('#knowledge-dialog').showModal(); $('#knowledge-status').textContent = 'Loading your local files…';
  try { await loadKnowledge(); } catch (error) { $('#knowledge-status').textContent = error.message; }
});
const openReminders = async () => {
  $('#reminder-dialog').showModal(); await loadReminders();
};
$('#show-reminders').addEventListener('click', openReminders);
$('#show-reminders-mobile').addEventListener('click', openReminders);
$('#reminder-refresh').addEventListener('click', loadReminders);
$('#reminder-form').addEventListener('submit', async event => {
  event.preventDefault();
  const objective = $('#reminder-objective').value.trim();
  const localTime = $('#reminder-at').value;
  const button = $('#reminder-save'); button.disabled = true;
  try {
    const response = await fetch('/tasks', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({objective, scope: 'User-created reminder; no external action is authorized.', remind_at: new Date(localTime).toISOString()}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Could not save this reminder.');
    $('#reminder-objective').value = ''; $('#reminder-at').value = '';
    await loadReminders();
  } catch (error) { $('#reminder-status').textContent = error.message; }
  finally { button.disabled = false; }
});
$('#knowledge-form').addEventListener('submit', async event => {
  event.preventDefault(); const input = $('#knowledge-file'); const file = input.files?.[0]; if (!file) return;
  const button = $('#knowledge-add'); button.disabled = true; button.textContent = 'Indexing…'; $('#knowledge-status').textContent = `Adding ${file.name}…`;
  try {
    const form = new FormData(); form.append('file', file);
    const response = await fetch('/knowledge', {method: 'POST', body: form}); const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Could not add this file.');
    input.value = ''; await loadKnowledge(); await loadHealth();
  } catch (error) { $('#knowledge-status').textContent = error.message; }
  finally { button.disabled = false; button.textContent = 'Index file'; }
});

async function loadAdminOverview() {
  const status = $('#admin-status'); const overview = $('#admin-overview');
  status.textContent = 'Loading Diya status…'; overview.replaceChildren();
  try {
    const response = await fetch('/admin/overview');
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Could not load admin status.');
    $('#admin-owner').textContent = data.owner;
    const entries = [
      ['Model', data.model],
      ['Knowledge pieces', data.indexed_chunks],
      ['Saved memories', data.memories],
      ['Conversations', data.conversations],
      ['Storage', data.storage],
      ['Voice output', data.voice_output],
      ['Voice input', data.voice_input],
    ];
    for (const [label, value] of entries) {
      const item = document.createElement('div'); item.className = 'admin-item';
      const name = document.createElement('span'); name.textContent = label;
      const detail = document.createElement('strong'); detail.textContent = value ?? 'Unavailable';
      item.append(name, detail); overview.append(item);
    }
    status.textContent = 'Diya is running locally. Only the signed-in admin can access these controls.';
  } catch (error) { status.textContent = error.message; }
}
$('#show-admin').addEventListener('click', async () => {
  $('#admin-dialog').showModal(); await loadAdminOverview();
});
$('#admin-refresh').addEventListener('click', loadAdminOverview);
$('#admin-memories').addEventListener('click', async () => {
  $('#admin-dialog').close();
  try { await loadMemories(); $('#memory-dialog').showModal(); }
  catch { setStatus('Could not load saved memories', 'Check the local API', 'error'); }
});
$('#memory-form').addEventListener('submit', async event => {
  event.preventDefault(); const input = $('#memory-text');
  try { const response = await fetch('/memory', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:input.value,kind:'fact'})}); const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Could not save this memory.'); input.value = ''; await loadMemories(); }
  catch (error) { const note = document.createElement('p'); note.className = 'error-banner'; note.textContent = error.message; $('#memory-list').prepend(note); }
});

async function boot() {
  await loadHealth();
  if (health?.mongo_ready) {
    try {
      const response = await fetch('/conversations'); const data = await response.json(); renderRecent(data.conversations || []);
      if (data.conversations?.length) await openConversation(data.conversations[0].id);
      else await startConversation();
    } catch { setStatus('MongoDB unavailable', 'Chats will not be saved until it reconnects', 'error'); }
  } else {
    try { const saved = JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); if (saved?.messages?.length) { activeConversation = saved.id; messages = saved.messages; for (const item of messages) renderMessage(item); } }
    catch { /* Ignore corrupt or blocked browser storage. */ }
  }
}
boot();
checkCaretakerNudges();
window.setInterval(checkCaretakerNudges, 30000);
document.addEventListener('visibilitychange', checkCaretakerNudges);
