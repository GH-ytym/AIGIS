const WUHAN_CENTER = [30.5928, 114.3055];

const map = L.map("map", {
	zoomControl: true,
	minZoom: 3,
	maxZoom: 19,
}).setView(WUHAN_CENTER, 12);

// Tencent tile y is reversed in XYZ coordinates.
const TencentTileLayer = L.TileLayer.extend({
	getTileUrl(coords) {
		const x = coords.x;
		const y = (1 << coords.z) - coords.y - 1;
		const s = this._getSubdomain(coords);
		const styleId = this.options.styleId ?? 0;
		const version = this.options.version ?? 347;
		return `https://rt${s}.map.gtimg.com/tile?z=${coords.z}&x=${x}&y=${y}&styleid=${styleId}&version=${version}`;
	},
});

new TencentTileLayer("", {
	subdomains: ["0", "1", "2", "3"],
	styleId: 0,
	version: 347,
	attribution: "&copy; 腾讯地图",
}).addTo(map);

L.marker(WUHAN_CENTER).addTo(map).bindPopup("武汉市中心").openPopup();

const inputEl = document.getElementById("nlInput");
const sendBtn = document.getElementById("sendBtn");
const statusBar = document.getElementById("statusBar");
const chatMessagesEl = document.getElementById("chatMessages");
const thinkingListEl = document.getElementById("thinkingList");
const toolDecisionListEl = document.getElementById("toolDecisionList");
const metaInfoEl = document.getElementById("metaInfo");

function setStatus(message, isError = false) {
	statusBar.textContent = message;
	statusBar.classList.toggle("error", isError);
}

function clearElement(el) {
	while (el.firstChild) {
		el.removeChild(el.firstChild);
	}
}

function normalizeStepText(step) {
	const raw = String(step || "").trim();
	return raw.replace(/^\d+[\.、:：]\s*/u, "").trim() || raw;
}

function setSingleTurnConversation(userText, assistantText) {
	clearElement(chatMessagesEl);

	const userRow = document.createElement("div");
	userRow.className = "message-row user";
	const userBubble = document.createElement("div");
	userBubble.className = "bubble";
	userBubble.textContent = userText;
	userRow.appendChild(userBubble);

	const assistantRow = document.createElement("div");
	assistantRow.className = "message-row assistant";
	const assistantBubble = document.createElement("div");
	assistantBubble.className = "bubble";
	assistantBubble.textContent = assistantText;
	assistantRow.appendChild(assistantBubble);

	chatMessagesEl.appendChild(userRow);
	chatMessagesEl.appendChild(assistantRow);
	chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

function fillOrderedList(targetEl, values, emptyText) {
	clearElement(targetEl);

	const items = Array.isArray(values) ? values : [];
	if (!items.length) {
		const item = document.createElement("li");
		item.textContent = emptyText;
		targetEl.appendChild(item);
		return;
	}

	for (const value of items) {
		const item = document.createElement("li");
		item.textContent = normalizeStepText(value);
		targetEl.appendChild(item);
	}
}

function renderMetaInfo(data) {
	const tools = Array.isArray(data.available_mcp_tools) ? data.available_mcp_tools : [];
	const line = `可用 MCP 工具: ${tools.length} 个`;
	metaInfoEl.textContent = line;
}

function composeAssistantReply(data) {
	if (!data.ok) {
		const reason = String(data.error || data.detail || "未知错误");
		return `这次没有规划成功。\n原因: ${reason}`;
	}

	const tasks = Array.isArray(data.llm_task_order) ? data.llm_task_order : [];
	const tools = Array.isArray(data.mcp_tool_order) ? data.mcp_tool_order : [];
	const searchPreview = String(data.search_result_preview || "").trim();
	const routePreview = String(data.route_result_preview || "").trim();
	if (!tasks.length && !tools.length) {
		return "我已收到你的请求，但这次没有产出有效的执行步骤。";
	}

	const previewLines = [];
	if (searchPreview) {
		previewLines.push(`搜索结果摘要: ${searchPreview}`);
	}
	if (routePreview) {
		previewLines.push(`路径结果摘要: ${routePreview}`);
	}

	if (tools.length) {
		const header = `我已完成本轮执行。\n本轮已调用: ${tools.join(" -> ")}`;
		if (previewLines.length) {
			return `${header}\n${previewLines.join("\n")}`;
		}
		return header;
	}

	return "我已完成本轮执行，下面是任务与决策结果。";
}

function renderSingleTurnResult(userText, data) {
	const assistantText = composeAssistantReply(data);
	setSingleTurnConversation(userText, assistantText);
	fillOrderedList(thinkingListEl, data.llm_task_order, "本轮未产出思考步骤");
	fillOrderedList(toolDecisionListEl, data.mcp_tool_order, "本轮未产出工具调用决策");
	renderMetaInfo(data);
}

function resetPanelsForNewRound() {
	clearElement(chatMessagesEl);
	const waiting = document.createElement("div");
	waiting.className = "empty-state";
	waiting.textContent = "消息已发送，LLM 正在思考中...";
	chatMessagesEl.appendChild(waiting);

	fillOrderedList(thinkingListEl, [], "等待思考结果...");
	fillOrderedList(toolDecisionListEl, [], "等待决策结果...");
	metaInfoEl.textContent = "";
}

async function sendMessageToBackend() {
	const text = String(inputEl.value || "").trim();
	if (!text) {
		return;
	}

	sendBtn.disabled = true;
	resetPanelsForNewRound();
	setStatus("正在请求后端，请稍候...", false);

	try {
		const response = await fetch("/api/message", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ message: text }),
		});

		if (!response.ok) {
			throw new Error(`HTTP ${response.status}`);
		}

		const data = await response.json();
		renderSingleTurnResult(text, data);
		setStatus(data.detail || "消息处理完成。", !data.ok);
		inputEl.value = "";
	} catch (error) {
		setSingleTurnConversation(text, `发送失败：${error}`);
		fillOrderedList(thinkingListEl, [], "请求失败，未获取思考过程");
		fillOrderedList(toolDecisionListEl, [], "请求失败，未获取调用决策");
		metaInfoEl.textContent = "";
		setStatus(`发送失败：${error}`, true);
	} finally {
		sendBtn.disabled = false;
	}
}

inputEl.addEventListener("keydown", (event) => {
	if (event.key === "Enter") {
		event.preventDefault();
		sendMessageToBackend();
	}
});

sendBtn.addEventListener("click", () => {
	sendMessageToBackend();
});

setStatus("页面已就绪：输入一句话开始单轮对话。", false);
fillOrderedList(thinkingListEl, [], "尚无思考过程");
fillOrderedList(toolDecisionListEl, [], "尚无调用决策");
