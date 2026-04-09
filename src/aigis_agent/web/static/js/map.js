const wuhanCenter = [30.5928, 114.3055];
const map = L.map("map").setView(wuhanCenter, 11);

L.tileLayer("https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}", {
  subdomains: ["1", "2", "3", "4"],
  maxZoom: 18,
  attribution: "&copy; 高德地图",
}).addTo(map);

const poiSearchInput = document.getElementById("poiSearchInput");
const poiSearchBtn = document.getElementById("poiSearchBtn");
const poiResults = document.getElementById("poiResults");
const poiDebugPanel = document.getElementById("poiDebugPanel");
const citySwitchPrompt = document.getElementById("citySwitchPrompt");
const citySwitchPromptText = document.getElementById("citySwitchPromptText");
const citySwitchYesBtn = document.getElementById("citySwitchYesBtn");
const citySwitchNoBtn = document.getElementById("citySwitchNoBtn");
const resultToggleBtn = document.getElementById("resultToggleBtn");
const cityPickerBtn = document.getElementById("cityPickerBtn");
const cityPickerPanel = document.getElementById("cityPickerPanel");
const provinceSelect = document.getElementById("provinceSelect");
const citySelect = document.getElementById("citySelect");
const districtSelect = document.getElementById("districtSelect");
const cityApplyBtn = document.getElementById("cityApplyBtn");
const cityPickerTip = document.getElementById("cityPickerTip");
const agentChatBubble = document.getElementById("agentChatBubble");
const SEARCH_RESULT_LIMIT = 20;
const DEFAULT_PROVINCE_NAME = "湖北省";
const DEFAULT_CITY_NAME = "武汉市";
const DEFAULT_DISTRICT_NAME = "洪山区";
let markers = [];
let markerByResultIndex = [];
let resultCards = [];
let currentResultItems = [];
let activeResultIndex = -1;
let isResultCollapsed = false;
let cityFocusMarker = null;
let provinceItems = [];
let cityItems = [];
let districtItems = [];
let citySwitchPromptResolver = null;
let agentBubbleLines = [];

function setResultCollapsed(collapsed) {
  isResultCollapsed = collapsed;
  poiResults.classList.toggle("collapsed", collapsed);

  if (!resultToggleBtn) {
    return;
  }

  resultToggleBtn.textContent = collapsed ? "展开结果" : "收起结果";
  resultToggleBtn.setAttribute("aria-expanded", String(!collapsed));
}

function setCityPickerTip(text, isError = false) {
  if (!cityPickerTip) {
    return;
  }

  cityPickerTip.textContent = text;
  cityPickerTip.classList.toggle("error", isError);
}

function setCityPickerOpen(open) {
  if (!cityPickerPanel || !cityPickerBtn) {
    return;
  }

  cityPickerPanel.classList.toggle("hidden", !open);
  cityPickerBtn.setAttribute("aria-expanded", String(open));
}

function setCitySwitchPromptOpen(open) {
  if (!citySwitchPrompt) {
    return;
  }

  citySwitchPrompt.classList.toggle("hidden", !open);
}

function resolveCitySwitchPrompt(accepted) {
  setCitySwitchPromptOpen(false);
  if (citySwitchYesBtn) {
    citySwitchYesBtn.onclick = null;
  }
  if (citySwitchNoBtn) {
    citySwitchNoBtn.onclick = null;
  }

  if (citySwitchPromptResolver) {
    const resolver = citySwitchPromptResolver;
    citySwitchPromptResolver = null;
    resolver(Boolean(accepted));
  }
}

function askCitySwitchConfirm(targetCity) {
  const safeTargetCity = String(targetCity || "").trim();
  if (!safeTargetCity) {
    return Promise.resolve(false);
  }

  const targetScope = splitCityDistrictHint(safeTargetCity);
  const currentScope = splitCityDistrictHint(getCurrentCityHintForSearch());
  const sameCity =
    normalizeDistrictName(targetScope.city || "") &&
    normalizeDistrictName(currentScope.city || "") &&
    normalizeDistrictName(targetScope.city || "") === normalizeDistrictName(currentScope.city || "");
  const promptText =
    sameCity && targetScope.district
      ? `是否要切换区县到${targetScope.district}？`
      : `是否要切换区域到${safeTargetCity}？`;

  if (!citySwitchPrompt || !citySwitchPromptText || !citySwitchYesBtn || !citySwitchNoBtn) {
    return Promise.resolve(window.confirm(promptText));
  }

  if (citySwitchPromptResolver) {
    resolveCitySwitchPrompt(false);
  }

  citySwitchPromptText.textContent = promptText;
  setCitySwitchPromptOpen(true);

  return new Promise((resolve) => {
    citySwitchPromptResolver = resolve;
    citySwitchYesBtn.onclick = () => {
      resolveCitySwitchPrompt(true);
    };
    citySwitchNoBtn.onclick = () => {
      resolveCitySwitchPrompt(false);
    };
  });
}

function getCitySwitchSuggestion(debugPayload, fallbackFromCity) {
  if (!debugPayload || typeof debugPayload !== "object") {
    return null;
  }

  const suggested = Boolean(debugPayload.city_switch_suggested);
  if (!suggested) {
    return null;
  }

  const targetCity = String(debugPayload.city_switch_to || "").trim();
  if (!targetCity) {
    return null;
  }

  const fromCity = String(
    debugPayload.city_switch_from || debugPayload.original_city_hint || fallbackFromCity || ""
  ).trim();

  const fromScope = splitCityDistrictHint(fromCity);
  const targetScope = splitCityDistrictHint(targetCity);
  const fromCityKey = normalizeDistrictName(fromScope.city || "");
  const toCityKey = normalizeDistrictName(targetScope.city || "");
  if (fromCityKey && toCityKey && fromCityKey !== toCityKey) {
    return null;
  }

  const fromDistrictKey = normalizeDistrictName(fromScope.district || "");
  const toDistrictKey = normalizeDistrictName(targetScope.district || "");
  if (fromDistrictKey && toDistrictKey && fromDistrictKey === toDistrictKey) {
    return null;
  }

  const fromKey = normalizeDistrictName(fromCity);
  const toKey = normalizeDistrictName(targetCity);
  if (fromKey && toKey && fromKey === toKey) {
    return null;
  }

  return {
    fromCity,
    targetCity,
  };
}

function getCitySwitchSuggestionFromResults(items, currentCityHint) {
  if (!Array.isArray(items) || !items.length) {
    return null;
  }

  const fromCity = canonicalizeCityHint(currentCityHint || "");
  const fromScope = splitCityDistrictHint(fromCity);

  let candidate = items[0] && typeof items[0] === "object" ? items[0] : null;
  if (fromScope.city) {
    const fromCityKey = normalizeDistrictName(fromScope.city);
    const sameCityCandidates = items.filter((item) => {
      if (!item || typeof item !== "object") {
        return false;
      }
      const itemCity = canonicalizeCityHint(item.city || item.province || "");
      return normalizeDistrictName(itemCity) === fromCityKey;
    });

    if (!sameCityCandidates.length) {
      return null;
    }

    if (fromScope.district) {
      const fromDistrictKey = normalizeDistrictName(fromScope.district);
      const diffDistrict = sameCityCandidates.find((item) => {
        const itemDistrict = canonicalizeCityHint(item.district || "");
        return normalizeDistrictName(itemDistrict) && normalizeDistrictName(itemDistrict) !== fromDistrictKey;
      });
      candidate = diffDistrict || sameCityCandidates[0];
    } else {
      candidate = sameCityCandidates[0];
    }
  }

  if (!candidate) {
    return null;
  }

  const targetCity = composeCityHint(candidate.city || candidate.province || "", candidate.district || "");
  if (!fromCity || !targetCity) {
    return null;
  }

  const fromKey = normalizeDistrictName(fromCity);
  const toKey = normalizeDistrictName(targetCity);
  if (!fromKey || !toKey || fromKey === toKey) {
    return null;
  }

  return {
    fromCity,
    targetCity,
  };
}

function renderAgentBubble(content, isError = false) {
  if (!agentChatBubble) {
    return;
  }

  const safeContent = String(content || "").trim();
  if (!safeContent) {
    agentChatBubble.classList.add("is-hidden");
    agentChatBubble.textContent = "";
    agentChatBubble.classList.remove("error");
    return;
  }

  agentChatBubble.textContent = safeContent;
  agentChatBubble.classList.remove("is-hidden");
  agentChatBubble.classList.toggle("error", isError);
  agentChatBubble.scrollTop = agentChatBubble.scrollHeight;
}

function resetAgentBubble() {
  agentBubbleLines = [];
  renderAgentBubble("");
}

function appendAgentBubbleLine(line, isError = false) {
  const text = String(line || "").trim();
  if (!text) {
    return;
  }

  agentBubbleLines.push(text);
  if (agentBubbleLines.length > 40) {
    agentBubbleLines = agentBubbleLines.slice(-40);
  }
  renderAgentBubble(agentBubbleLines.join("\n"), isError);
}

function clearMarkers() {
  markers.forEach((m) => map.removeLayer(m));
  markers = [];
  markerByResultIndex = [];
}

function setActiveResultIndex(index) {
  activeResultIndex = Number.isInteger(index) ? index : -1;

  resultCards.forEach((card, idx) => {
    card.classList.toggle("active", idx === activeResultIndex);
  });

  markerByResultIndex.forEach((marker, idx) => {
    if (!marker) {
      return;
    }
    marker.setZIndexOffset(idx === activeResultIndex ? 1200 : 0);
  });
}

function focusResultByIndex(index, options = {}) {
  if (!Number.isInteger(index) || index < 0 || index >= currentResultItems.length) {
    return;
  }

  const item = currentResultItems[index] || {};
  const marker = markerByResultIndex[index] || null;
  setActiveResultIndex(index);

  if (marker) {
    marker.openPopup();
  }

  const canMoveMap =
    options.moveMap !== false && typeof item.lat === "number" && typeof item.lon === "number";
  if (canMoveMap) {
    const nextZoom = Math.max(map.getZoom(), 15);
    map.flyTo([item.lat, item.lon], nextZoom, { duration: 0.45 });
  }

  if (options.scrollCard !== false) {
    const card = resultCards[index];
    if (card) {
      card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }
}

function renderStatusCard(text, isError = false) {
  currentResultItems = [];
  resultCards = [];
  setActiveResultIndex(-1);

  const status = document.createElement("div");
  status.className = `result-card status-card${isError ? " error" : ""}`;
  status.textContent = text;

  poiResults.innerHTML = "";
  poiResults.appendChild(status);
}

function renderPoiDebug(debugPayload) {
  if (!poiDebugPanel) {
    return;
  }

  if (!debugPayload || typeof debugPayload !== "object") {
    poiDebugPanel.textContent = "调试信息：等待搜索...";
    return;
  }

  if (typeof debugPayload.status === "string") {
    poiDebugPanel.textContent = `调试信息：${debugPayload.status}`;
    return;
  }

  if (typeof debugPayload.error === "string") {
    poiDebugPanel.textContent = `调试信息：${debugPayload.error}`;
    return;
  }

  const correctedItems = Array.isArray(debugPayload.corrected_items)
    ? debugPayload.corrected_items.map((item) => ({
        name: item.name || "",
        address: item.address || "",
        lat: item.lat,
        lon: item.lon,
      }))
    : [];

  const agentMeta =
    debugPayload.agent && typeof debugPayload.agent === "object"
      ? {
          selected_tool: debugPayload.agent.selected_tool || null,
          tasks: Array.isArray(debugPayload.agent.tasks) ? debugPayload.agent.tasks : [],
          message: debugPayload.agent.message || null,
          fallback_used: Boolean(debugPayload.agent.fallback_used),
        }
      : null;

  const view = {
    ai_refine_triggered: Boolean(debugPayload.ai_refine_triggered),
    ai_refine_applied: Boolean(debugPayload.ai_refine_applied),
    city_switch_suggested: Boolean(debugPayload.city_switch_suggested),
    city_switch_applied: Boolean(debugPayload.city_switch_applied),
    city_switch_from: debugPayload.city_switch_from || null,
    city_switch_to: debugPayload.city_switch_to || null,
    query_before: String(debugPayload.original_query || ""),
    query_after: String(
      debugPayload.corrected_query || debugPayload.final_query || debugPayload.original_query || ""
    ),
    final_query: String(debugPayload.final_query || ""),
    city_before: debugPayload.original_city_hint || null,
    city_after: debugPayload.final_city_hint || null,
    trigger_reason: debugPayload.trigger_reason || null,
    corrected_result_count: Number(debugPayload.corrected_result_count || 0),
    corrected_results: correctedItems,
    agent: agentMeta,
  };

  poiDebugPanel.textContent = JSON.stringify(view, null, 2);
}

function renderPoiResults(items, query) {
  currentResultItems = Array.isArray(items) ? items : [];
  resultCards = [];
  setActiveResultIndex(-1);

  poiResults.innerHTML = "";

  if (!currentResultItems.length) {
    renderStatusCard(`未找到“${query}”相关结果，可尝试增加城市前缀。`);
    return;
  }

  const fragment = document.createDocumentFragment();
  currentResultItems.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "result-card";
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `定位到第${index + 1}个结果：${item.name || query}`);

    const title = document.createElement("h3");
    title.className = "result-title";
    title.textContent = item.name || query;

    const address = document.createElement("p");
    address.className = "result-address";
    address.textContent = item.address || "未提供详细地址";

    card.appendChild(title);
    card.appendChild(address);

    card.addEventListener("click", () => {
      focusResultByIndex(index);
    });
    card.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") {
        return;
      }
      event.preventDefault();
      focusResultByIndex(index);
    });

    resultCards.push(card);
    fragment.appendChild(card);
  });

  poiResults.appendChild(fragment);
}

function plotMarkers(items) {
  clearMarkers();
  markerByResultIndex = new Array(items.length).fill(null);
  if (!items.length) {
    return;
  }

  const points = [];
  items.forEach((item, index) => {
    if (typeof item.lat !== "number" || typeof item.lon !== "number") {
      return;
    }

    const marker = L.marker([item.lat, item.lon])
      .addTo(map)
      .bindPopup(item.address || item.name || "未命名地点");
    markers.push(marker);
    markerByResultIndex[index] = marker;

    marker.on("click", () => {
      focusResultByIndex(index, { moveMap: false, scrollCard: true });
    });

    points.push([item.lat, item.lon]);
  });

  if (!points.length) {
    return;
  }

  if (points.length === 1) {
    map.setView(points[0], 14);
    return;
  }

  const bounds = L.latLngBounds(points);
  map.fitBounds(bounds, { padding: [34, 34], maxZoom: 15 });
}

async function parseJsonResponse(res) {
  const raw = await res.text();
  let payload = null;
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = null;
    }
  }

  if (!res.ok) {
    const detail =
      payload && typeof payload.detail === "string"
        ? payload.detail
        : raw.trim();
    throw new Error(detail ? `HTTP ${res.status}: ${detail}` : `HTTP ${res.status}`);
  }

  return payload ?? {};
}

async function getJson(url) {
  const res = await fetch(url, {
    method: "GET",
  });
  return parseJsonResponse(res);
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseJsonResponse(res);
}

async function postNdjsonStream(url, body, onEvent) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    await parseJsonResponse(res);
    return;
  }

  if (!res.body) {
    throw new Error("流式响应不可用");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }

      try {
        const event = JSON.parse(trimmed);
        onEvent(event);
      } catch {
        // Ignore malformed stream fragments and continue reading.
      }
    }
  }

  const tail = buffer.trim();
  if (tail) {
    try {
      const event = JSON.parse(tail);
      onEvent(event);
    } catch {
      // Ignore malformed tail chunk.
    }
  }
}

function formatTaskLine(task) {
  if (!task || typeof task !== "object") {
    return "";
  }

  const step = Number(task.step_id || 0);
  const tool = String(task.tool_name || "unknown").trim();
  const reason = String(task.reason || "").trim();
  if (!tool) {
    return "";
  }

  if (reason) {
    return `${step || "?"}. ${tool} - ${reason}`;
  }

  return `${step || "?"}. ${tool}`;
}

function formatToolResultBrief(toolName, result) {
  if (!result || typeof result !== "object") {
    return `Tool ${toolName} 已执行`;
  }

  if (result.status === "error") {
    return `Tool ${toolName} 执行失败: ${result.error || "未知错误"}`;
  }

  if (toolName === "search_poi") {
    const count = Number(result.count || 0);
    const topItems = Array.isArray(result.top_items) ? result.top_items : [];
    const firstName = topItems.length ? String(topItems[0].name || "").trim() : "";
    if (firstName) {
      return `Tool search_poi 返回 ${count} 条，首条: ${firstName}`;
    }
    return `Tool search_poi 返回 ${count} 条结果`;
  }

  return `Tool ${toolName} 执行完成`;
}

async function runAssistStream(payload) {
  let finalOutcome = null;

  await postNdjsonStream("/v1/agent/assist/stream", payload, (event) => {
    const eventType = String(event?.event || "").trim();
    if (!eventType) {
      return;
    }

    if (eventType === "status") {
      appendAgentBubbleLine(String(event.message || "助手处理中..."));
      return;
    }

    if (eventType === "planner_start") {
      appendAgentBubbleLine(String(event.message || "正在生成任务列表..."));
      return;
    }

    if (eventType === "planner_result") {
      appendAgentBubbleLine("任务列表已生成:");
      const tasks = Array.isArray(event.tasks) ? event.tasks : [];
      if (!tasks.length) {
        appendAgentBubbleLine("- 暂无任务，准备使用默认工具");
      } else {
        for (const task of tasks) {
          const line = formatTaskLine(task);
          if (line) {
            appendAgentBubbleLine(line);
          }
        }
      }
      if (event.fallback_used) {
        appendAgentBubbleLine("规划阶段触发了降级策略");
      }
      return;
    }

    if (eventType === "tool_start") {
      const toolName = String(event.tool_name || "unknown").trim();
      appendAgentBubbleLine(`开始调用 tool: ${toolName}`);
      return;
    }

    if (eventType === "tool_result") {
      const toolName = String(event.tool_name || "unknown").trim();
      appendAgentBubbleLine(formatToolResultBrief(toolName, event.result));
      return;
    }

    if (eventType === "done") {
      if (event.outcome && typeof event.outcome === "object") {
        finalOutcome = event.outcome;
        appendAgentBubbleLine(String(event.outcome.message || "任务执行完成"));
      } else {
        appendAgentBubbleLine("任务执行完成");
      }
    }
  });

  if (!finalOutcome) {
    throw new Error("助手未返回最终执行结果");
  }

  return finalOutcome;
}

async function fetchDistrictItems(keywords, filterAdcode = "") {
  const params = new URLSearchParams({
    keywords,
    subdistrict: "1",
    page: "1",
    offset: "50",
  });
  if (filterAdcode) {
    params.set("filter", filterAdcode);
  }

  const data = await getJson(`/v1/poi/districts?${params.toString()}`);
  return Array.isArray(data.items) ? data.items : [];
}

function buildSelectOptions(select, items, placeholder, formatLabel) {
  if (!select) {
    return;
  }

  select.innerHTML = "";
  const placeholderOption = document.createElement("option");
  placeholderOption.value = "";
  placeholderOption.textContent = placeholder;
  placeholderOption.disabled = true;
  placeholderOption.selected = true;
  select.appendChild(placeholderOption);

  items.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = typeof formatLabel === "function" ? formatLabel(item) : item.name;
    select.appendChild(option);
  });
}

function getSelectedItem(select, items) {
  if (!select) {
    return null;
  }

  const index = Number.parseInt(select.value, 10);
  if (Number.isNaN(index) || index < 0) {
    return null;
  }

  return items[index] ?? null;
}

function normalizeDistrictName(name) {
  return String(name || "")
    .trim()
    .replace(/\s+/g, "")
    .replace(/(\(直辖\)|（直辖）)/gu, "")
    .replace(/(省|市|区|县|自治区|自治州|地区|盟|特别行政区)$/u, "");
}

function canonicalizeCityHint(name) {
  const raw = String(name || "")
    .trim()
    .replace(/\s+/g, "")
    .replace(/(\(直辖\)|（直辖）)/gu, "");

  if (!raw) {
    return "";
  }

  if (/^(北京|上海|天津|重庆)$/u.test(raw)) {
    return `${raw}市`;
  }

  if (/(市|自治州|地区|盟|特别行政区|自治区|省|县|区|旗)$/u.test(raw)) {
    return raw;
  }

  return `${raw}市`;
}

function splitCityDistrictHint(name) {
  const raw = canonicalizeCityHint(name);
  if (!raw) {
    return {
      city: "",
      district: "",
    };
  }

  const explicit = raw.match(
    /^(.+?(?:市|自治州|地区|盟|特别行政区|自治区|省))(.+?(?:自治县|自治旗|特区|林区|区|县|旗))$/u
  );
  if (explicit) {
    return {
      city: canonicalizeCityHint(explicit[1]),
      district: canonicalizeCityHint(explicit[2]),
    };
  }

  const loose = raw.match(/^([\u4e00-\u9fff]{2,8}?)([\u4e00-\u9fff]{2,12}(?:自治县|自治旗|特区|林区|区|县|旗))$/u);
  if (loose && loose[1] && loose[2]) {
    return {
      city: canonicalizeCityHint(loose[1]),
      district: canonicalizeCityHint(loose[2]),
    };
  }

  if (/(自治县|自治旗|特区|林区|区|县|旗)$/u.test(raw)) {
    return {
      city: "",
      district: raw,
    };
  }

  return {
    city: raw,
    district: "",
  };
}

function composeCityHint(cityName, districtName = "") {
  const city = canonicalizeCityHint(cityName);
  const district = canonicalizeCityHint(districtName);

  if (city && district) {
    const cityAlias = normalizeDistrictName(city);
    const districtAlias = normalizeDistrictName(district);
    if (districtAlias && cityAlias && districtAlias.startsWith(cityAlias)) {
      return district;
    }
    return `${city}${district}`;
  }

  return district || city;
}

function extractScopeHintFromQuery(query) {
  const raw = String(query || "")
    .trim()
    .replace(/\s+/g, "")
    .replace(/[，。！？、；：,.!?;:]/gu, "");

  if (!raw) {
    return "";
  }

  const stripped = raw.replace(
    /^(?:帮我|麻烦帮我|麻烦|请|请帮我|给我|找一找|找一下|找下|查一下|查下|搜一下|搜下|看下|看看|帮忙|帮我在|在)+/u,
    ""
  );
  if (!stripped) {
    return "";
  }

  const districtPrefix = stripped.match(/^([\u4e00-\u9fff]{2,14}(?:自治县|自治旗|特区|林区|区|县|旗))/u);
  if (districtPrefix && districtPrefix[1]) {
    const scope = splitCityDistrictHint(districtPrefix[1]);
    const composed = composeCityHint(scope.city, scope.district);
    if (composed) {
      return composed;
    }
  }

  const cityPrefix = stripped.match(/^([\u4e00-\u9fff]{2,12}(?:市|自治州|地区|盟|特别行政区|自治区|省))/u);
  if (cityPrefix && cityPrefix[1]) {
    const scope = splitCityDistrictHint(cityPrefix[1]);
    const composed = composeCityHint(scope.city, scope.district);
    if (composed) {
      return composed;
    }
  }

  return "";
}

function findItemIndexByName(items, preferredName) {
  if (!preferredName || !items.length) {
    return -1;
  }

  const normalizedPreferred = normalizeDistrictName(preferredName);
  return items.findIndex((item) => {
    const raw = String(item.name || "").trim();
    if (raw === preferredName) {
      return true;
    }
    const normalizedRaw = normalizeDistrictName(raw);
    return normalizedRaw && normalizedRaw === normalizedPreferred;
  });
}

function getCurrentCityItem() {
  const selectedDistrict = getSelectedItem(districtSelect, districtItems);
  if (selectedDistrict) {
    return selectedDistrict;
  }

  const selectedCity = getSelectedItem(citySelect, cityItems);
  if (selectedCity) {
    return selectedCity;
  }

  return getSelectedItem(provinceSelect, provinceItems);
}

function getCurrentCityHintForSearch() {
  const selectedCity = getSelectedItem(citySelect, cityItems);
  const selectedDistrict = getSelectedItem(districtSelect, districtItems);
  if (selectedDistrict && selectedDistrict.name) {
    return composeCityHint(selectedCity && selectedCity.name ? selectedCity.name : "", selectedDistrict.name);
  }

  if (selectedCity && selectedCity.name) {
    return canonicalizeCityHint(selectedCity.name);
  }

  const selectedProvince = getSelectedItem(provinceSelect, provinceItems);
  if (selectedProvince && selectedProvince.name) {
    return canonicalizeCityHint(selectedProvince.name);
  }

  return "";
}

function inferCityHintForQuery(query) {
  const explicitScopeHint = extractScopeHintFromQuery(query);
  if (explicitScopeHint) {
    return explicitScopeHint;
  }

  const currentCity = getCurrentCityItem();
  if (!currentCity || !currentCity.name) {
    return null;
  }

  const rawQuery = String(query || "").trim();
  if (!rawQuery) {
    return null;
  }

  const normalizedQuery = normalizeDistrictName(rawQuery);
  const cityName = String(currentCity.name || "").trim();
  const cityAlias = normalizeDistrictName(cityName);

  if (rawQuery.includes(cityName) || (cityAlias && normalizedQuery.includes(cityAlias))) {
    return null;
  }

  const hasExplicitAdministrativeText =
    /(省|市|自治区|自治州|地区|盟|特别行政区)/u.test(rawQuery) ||
    /(北京|上海|天津|重庆)/u.test(rawQuery);
  if (hasExplicitAdministrativeText) {
    return null;
  }

  const knownCityNames = cityItems.map((item) => String(item.name || ""));
  const containsKnownCityName = knownCityNames.some((name) => {
    const alias = normalizeDistrictName(name);
    return (name && rawQuery.includes(name)) || (alias && normalizedQuery.includes(alias));
  });
  if (containsKnownCityName) {
    return null;
  }

  const scopedHint = getCurrentCityHintForSearch();
  return scopedHint || canonicalizeCityHint(cityName);
}

function focusToDistrict(item) {
  if (!item || typeof item.center_lat !== "number" || typeof item.center_lon !== "number") {
    setCityPickerTip("所选区域缺少中心点坐标，无法定位。", true);
    return;
  }

  if (cityFocusMarker) {
    map.removeLayer(cityFocusMarker);
  }

  cityFocusMarker = L.marker([item.center_lat, item.center_lon])
    .addTo(map)
    .bindPopup(`已切换到 ${item.name}`);

  let zoom = 9;
  if (item.level === "city") {
    zoom = 10;
  }
  if (item.level === "district" || item.level === "street") {
    zoom = 12;
  }

  map.setView([item.center_lat, item.center_lon], zoom);
  cityFocusMarker.openPopup();
  setCityPickerTip(`已切换到 ${item.name}`);
  setCityPickerOpen(false);
}

async function onCityChange(preferredDistrictName = "") {
  const selectedCity = getSelectedItem(citySelect, cityItems);
  districtItems = [];
  buildSelectOptions(districtSelect, districtItems, "请选择区县");
  if (districtSelect) {
    districtSelect.disabled = true;
  }

  if (!selectedCity) {
    return;
  }

  setCityPickerTip("正在加载区县...");
  try {
    const children = await fetchDistrictItems(selectedCity.adcode);
    const counties = children.filter((item) => item.level === "district");
    const streets = children.filter((item) => item.level === "street");
    districtItems = counties.length ? counties : streets;

    if (!districtItems.length) {
      setCityPickerTip("当前城市暂无区县数据，可直接切换到该城市。", false);
      return;
    }

    buildSelectOptions(districtSelect, districtItems, "请选择区县");
    if (districtSelect) {
      districtSelect.disabled = false;
      const preferredIndex = findItemIndexByName(districtItems, preferredDistrictName);
      districtSelect.selectedIndex = preferredIndex >= 0 ? preferredIndex + 1 : 1;
    }
    setCityPickerTip("请选择区县并点击切换。", false);
  } catch (error) {
    setCityPickerTip(`加载区县失败: ${error}`, true);
  }
}

async function onProvinceChange(preferredCityName = "", preferredDistrictName = "") {
  const selectedProvince = getSelectedItem(provinceSelect, provinceItems);
  cityItems = [];
  districtItems = [];
  buildSelectOptions(citySelect, cityItems, "请选择地级市");
  buildSelectOptions(districtSelect, districtItems, "请选择区县");
  if (citySelect) {
    citySelect.disabled = true;
  }
  if (districtSelect) {
    districtSelect.disabled = true;
  }

  if (!selectedProvince) {
    return;
  }

  setCityPickerTip("正在加载地级市...");
  try {
    const children = await fetchDistrictItems(selectedProvince.adcode);
    const cities = children.filter((item) => item.level === "city");

    if (cities.length) {
      cityItems = cities;
      buildSelectOptions(citySelect, cityItems, "请选择地级市");
      if (citySelect) {
        citySelect.disabled = false;
        const preferredIndex = findItemIndexByName(cityItems, preferredCityName);
        citySelect.selectedIndex = preferredIndex >= 0 ? preferredIndex + 1 : 1;
      }
      await onCityChange(preferredDistrictName);
      return;
    }

    const counties = children.filter((item) => item.level === "district");
    if (counties.length) {
      cityItems = [
        {
          name: `${selectedProvince.name}（直辖）`,
          adcode: selectedProvince.adcode,
          level: "city",
          center_lat: selectedProvince.center_lat,
          center_lon: selectedProvince.center_lon,
        },
      ];
      buildSelectOptions(citySelect, cityItems, "请选择地级市", (item) => item.name);
      if (citySelect) {
        citySelect.disabled = false;
        citySelect.selectedIndex = 1;
      }

      districtItems = counties;
      buildSelectOptions(districtSelect, districtItems, "请选择区县");
      if (districtSelect) {
        districtSelect.disabled = false;
        const preferredIndex = findItemIndexByName(districtItems, preferredDistrictName);
        districtSelect.selectedIndex = preferredIndex >= 0 ? preferredIndex + 1 : 1;
      }
      setCityPickerTip("请选择区县并点击切换。", false);
      return;
    }

    setCityPickerTip("当前省份暂无可用区县数据。", true);
  } catch (error) {
    setCityPickerTip(`加载地级市失败: ${error}`, true);
  }
}

async function loadProvinces() {
  if (!provinceSelect || !citySelect || !districtSelect) {
    return;
  }

  provinceSelect.disabled = true;
  citySelect.disabled = true;
  districtSelect.disabled = true;
  buildSelectOptions(provinceSelect, [], "加载中...");
  buildSelectOptions(citySelect, [], "请选择地级市");
  buildSelectOptions(districtSelect, [], "请选择区县");

  setCityPickerTip("正在加载省份...");
  try {
    const rawItems = await fetchDistrictItems("中国");
    provinceItems = rawItems.filter((item) => item.level === "province" || item.level === "city");

    if (!provinceItems.length) {
      setCityPickerTip("未获取到省级数据。", true);
      return;
    }

    buildSelectOptions(provinceSelect, provinceItems, "请选择省/直辖市");
    provinceSelect.disabled = false;
    const defaultProvinceIndex = findItemIndexByName(provinceItems, DEFAULT_PROVINCE_NAME);
    provinceSelect.selectedIndex = defaultProvinceIndex >= 0 ? defaultProvinceIndex + 1 : 1;
    await onProvinceChange(DEFAULT_CITY_NAME, DEFAULT_DISTRICT_NAME);
    applyCitySelection();
  } catch (error) {
    setCityPickerTip(`加载省份失败: ${error}`, true);
  }
}

function applyCitySelection() {
  const selectedDistrict = getSelectedItem(districtSelect, districtItems);
  if (selectedDistrict) {
    focusToDistrict(selectedDistrict);
    return;
  }

  const selectedCity = getSelectedItem(citySelect, cityItems);
  if (selectedCity) {
    focusToDistrict(selectedCity);
    return;
  }

  const selectedProvince = getSelectedItem(provinceSelect, provinceItems);
  if (selectedProvince) {
    focusToDistrict(selectedProvince);
    return;
  }

  setCityPickerTip("请先选择区域。", true);
}

async function syncCityPickerByTopResult(topItem = null, fallbackCityName = "") {
  if (!provinceSelect || !citySelect || !districtSelect) {
    return;
  }

  if (!provinceItems.length) {
    await loadProvinces();
  }

  const safeTopItem = topItem && typeof topItem === "object" ? topItem : {};
  const fallbackScope = splitCityDistrictHint(fallbackCityName);
  const targetProvince = String(safeTopItem.province || "").trim();
  const targetCity = canonicalizeCityHint(safeTopItem.city || fallbackScope.city || fallbackCityName || "");
  const targetDistrict = String(safeTopItem.district || fallbackScope.district || "").trim();

  if (!targetCity) {
    return;
  }

  let provinceIndex = findItemIndexByName(provinceItems, targetProvince);
  if (provinceIndex < 0) {
    provinceIndex = findItemIndexByName(provinceItems, targetCity);
  }

  if (provinceIndex >= 0) {
    provinceSelect.selectedIndex = provinceIndex + 1;
    await onProvinceChange(targetCity, targetDistrict);
  } else {
    for (let idx = 0; idx < provinceItems.length; idx += 1) {
      provinceSelect.selectedIndex = idx + 1;
      await onProvinceChange(targetCity, targetDistrict);
      const cityIndex = findItemIndexByName(cityItems, targetCity);
      if (cityIndex >= 0) {
        provinceIndex = idx;
        break;
      }
    }
  }

  if (!citySelect.disabled) {
    const cityIndex = findItemIndexByName(cityItems, targetCity);
    if (cityIndex >= 0) {
      citySelect.selectedIndex = cityIndex + 1;
      await onCityChange(targetDistrict);
    }
  }

  if (!districtSelect.disabled) {
    const districtIndex = findItemIndexByName(districtItems, targetDistrict);
    if (districtIndex >= 0) {
      districtSelect.selectedIndex = districtIndex + 1;
    }
  }

  applyCitySelection();
}

function initCityPicker() {
  if (!cityPickerBtn || !cityPickerPanel) {
    return;
  }

  cityPickerBtn.addEventListener("click", async () => {
    const shouldOpen = cityPickerPanel.classList.contains("hidden");
    setCityPickerOpen(shouldOpen);
    if (shouldOpen && !provinceItems.length) {
      await loadProvinces();
    }
  });

  if (provinceSelect) {
    provinceSelect.addEventListener("change", () => {
      onProvinceChange();
    });
  }
  if (citySelect) {
    citySelect.addEventListener("change", () => {
      onCityChange();
    });
  }
  if (cityApplyBtn) {
    cityApplyBtn.addEventListener("click", applyCitySelection);
  }

  document.addEventListener("click", (event) => {
    if (!cityPickerPanel || cityPickerPanel.classList.contains("hidden")) {
      return;
    }

    if (cityPickerPanel.contains(event.target) || cityPickerBtn.contains(event.target)) {
      return;
    }

    setCityPickerOpen(false);
  });

  setCityPickerTip("点击“选择城市”开始加载行政区划。", false);
  loadProvinces();
}

async function searchPoi() {
  resolveCitySwitchPrompt(false);
  resetAgentBubble();
  const query = poiSearchInput.value.trim();
  if (!query) {
    renderStatusCard("请输入查询文本");
    renderPoiDebug({ status: "请输入查询文本" });
    appendAgentBubbleLine("请输入查询文本");
    return;
  }

  setResultCollapsed(false);
  renderStatusCard("搜索中...");
  renderPoiDebug({ status: "搜索中..." });

  try {
    const inferredCityHint = inferCityHintForQuery(query);
    const currentCityHint = getCurrentCityHintForSearch();
    const cityHint = inferredCityHint || currentCityHint;
    const basePayload = {
      query,
      limit: SEARCH_RESULT_LIMIT,
      allow_city_switch: false,
    };
    if (cityHint) {
      basePayload.city_hint = cityHint;
    }

    let assistData = await runAssistStream(basePayload);
    const selectedTool = String(assistData.selected_tool || "").trim();
    const tasks = Array.isArray(assistData.tasks) ? assistData.tasks : [];
    const toolResult =
      assistData.tool_result && typeof assistData.tool_result === "object"
        ? assistData.tool_result
        : {};

    if (selectedTool !== "search_poi") {
      const message = String(assistData.message || "当前请求暂未映射为可展示的地图检索结果");
      renderStatusCard(message);
      renderPoiDebug({
        status: message,
        agent: {
          selected_tool: selectedTool || "unknown",
          tasks,
          fallback_used: Boolean(assistData.fallback_used),
        },
        tool_result: toolResult,
      });
      clearMarkers();
      return;
    }

    let data = toolResult;
    const debugForSwitch =
      data && typeof data.debug === "object" && data.debug !== null
        ? data.debug
        : {};
    const currentCityForSwitch = cityHint || getCurrentCityHintForSearch();
    let switchSuggestion = getCitySwitchSuggestion(debugForSwitch, currentCityForSwitch);
    if (!switchSuggestion) {
      switchSuggestion = getCitySwitchSuggestionFromResults(
        Array.isArray(data.items) ? data.items : [],
        currentCityForSwitch
      );
    }
    if (switchSuggestion) {
      const confirmed = await askCitySwitchConfirm(switchSuggestion.targetCity);
      if (confirmed) {
        await syncCityPickerByTopResult(null, switchSuggestion.targetCity);

        const switchedPayload = {
          query,
          limit: SEARCH_RESULT_LIMIT,
          allow_city_switch: true,
        };
        const switchedScope = splitCityDistrictHint(switchSuggestion.targetCity);
        const switchedCityHint = composeCityHint(
          switchedScope.city || switchSuggestion.targetCity,
          switchedScope.district
        );
        if (switchedCityHint) {
          switchedPayload.city_hint = switchedCityHint;
        } else if (cityHint) {
          switchedPayload.city_hint = cityHint;
        }
        assistData = await runAssistStream(switchedPayload);
        data =
          assistData.tool_result && typeof assistData.tool_result === "object"
            ? assistData.tool_result
            : {};

        const switchedItems = Array.isArray(data.items) ? data.items : [];
        if (switchedItems.length) {
          await syncCityPickerByTopResult(switchedItems[0], switchSuggestion.targetCity);
        }
      }
    }

    const items = Array.isArray(data.items) ? data.items : [];
    renderPoiResults(items, query);
    const debugPayload =
      data.debug && typeof data.debug === "object"
        ? {
            ...data.debug,
            agent: {
              selected_tool: String(assistData.selected_tool || "search_poi"),
              tasks: Array.isArray(assistData.tasks) ? assistData.tasks : [],
              message: String(assistData.message || ""),
              fallback_used: Boolean(assistData.fallback_used),
            },
          }
        : {
            status: "tool 未返回调试结构",
            agent: {
              selected_tool: String(assistData.selected_tool || "search_poi"),
              tasks: Array.isArray(assistData.tasks) ? assistData.tasks : [],
              message: String(assistData.message || ""),
              fallback_used: Boolean(assistData.fallback_used),
            },
          };
    renderPoiDebug(debugPayload);
    plotMarkers(items);
  } catch (error) {
    renderStatusCard(`查询失败: ${error}`, true);
    renderPoiDebug({ error: String(error) });
    appendAgentBubbleLine(`执行失败: ${error}`, true);
    clearMarkers();
  }
}

if (poiSearchBtn) {
  poiSearchBtn.addEventListener("click", searchPoi);
}

if (poiSearchInput) {
  poiSearchInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    searchPoi();
  });
}

if (resultToggleBtn) {
  resultToggleBtn.addEventListener("click", () => {
    setResultCollapsed(!isResultCollapsed);
  });
}

renderStatusCard("输入关键词后点击放大镜开始搜索。");
renderPoiDebug(null);
setResultCollapsed(false);
initCityPicker();
