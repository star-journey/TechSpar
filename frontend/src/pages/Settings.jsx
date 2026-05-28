import { useState, useEffect, useRef, useCallback } from "react";
import {
  Server,
  Sliders,
  Eye,
  EyeOff,
  Loader2,
  Check,
  Mic,
  Square,
  Trash2,
  Database,
  Download,
  Upload,
  AlertTriangle,
  Boxes,
  UserCog,
} from "lucide-react";
import { getSettings, updateSettings } from "../api/interview";
import {
  getVoiceprintStatus,
  putVoiceprintCredentials,
  enrollVoiceprint,
  deleteVoiceprintEnrollment,
} from "../api/voiceprint";
import { exportData, importData } from "../api/dataMigration";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";

// 录音参数
const VP_SAMPLE_RATE = 16000;
const VP_MIN_SECONDS = 6;

// ── WAV / PCM 工具（用于声纹录音上传）──

function encodeWav(pcm16, sampleRate) {
  const dataSize = pcm16.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeStr = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);
  let offset = 44;
  for (let i = 0; i < pcm16.length; i++) {
    view.setInt16(offset, pcm16[i], true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function mergeFloat32(chunks) {
  const total = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Float32Array(total);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

function resampleToPcm16(input, inputRate, outputRate) {
  if (inputRate === outputRate) {
    const pcm = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return pcm;
  }
  const ratio = inputRate / outputRate;
  const outLen = Math.max(1, Math.round(input.length / ratio));
  const pcm = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const src = i * ratio;
    const lo = Math.floor(src);
    const hi = Math.min(lo + 1, input.length - 1);
    const w = src - lo;
    const v = (input[lo] ?? 0) * (1 - w) + (input[hi] ?? 0) * w;
    const s = Math.max(-1, Math.min(1, v));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm;
}

const DIVERGENCE_OPTIONS = [
  { value: 1, label: "聚焦薄弱", description: "100% 针对存在弱点的知识域，适合考前专项突击" },
  { value: 2, label: "侧重薄弱", description: "约 70% 针对薄弱点，30% 拓展至新知识点" },
  { value: 3, label: "均衡", description: "薄弱环节巩固与全新知识盲区发掘各占 50%" },
  { value: 4, label: "侧重探索", description: "约 30% 回顾薄弱点，70% 探索全新知识层面" },
  { value: 5, label: "全面探索", description: "100% 探索未涉猎过的新知识领域，发掘潜在盲区" },
];

export default function Settings() {
  const [apiBase, setApiBase] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [numQuestions, setNumQuestions] = useState(10);
  const [divergence, setDivergence] = useState(3);
  const [showKey, setShowKey] = useState(false);

  // Embedding 配置（全局，hot-reload）
  const [embBackend, setEmbBackend] = useState("");  // "" | api | local
  const [embApiBase, setEmbApiBase] = useState("");
  const [embApiKey, setEmbApiKey] = useState("");
  const [embApiModel, setEmbApiModel] = useState("");
  const [embLocalModel, setEmbLocalModel] = useState("");
  const [embLocalPath, setEmbLocalPath] = useState("");
  const [showEmbKey, setShowEmbKey] = useState(false);

  // 账户/系统配置（全局）
  const [allowRegistration, setAllowRegistration] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("llm");

  // 声纹识别状态
  const [vpStatus, setVpStatus] = useState({ configured: false, enrolled: false });
  const [vpSecretId, setVpSecretId] = useState("");
  const [vpSecretKey, setVpSecretKey] = useState("");
  const [vpAppId, setVpAppId] = useState("");
  const [showVpKey, setShowVpKey] = useState(false);
  const [vpBusy, setVpBusy] = useState(false);
  const [vpMessage, setVpMessage] = useState("");
  const [vpRecording, setVpRecording] = useState(false);
  const [vpRecordingSec, setVpRecordingSec] = useState(0);

  const vpStreamRef = useRef(null);
  const vpCtxRef = useRef(null);
  const vpSourceRef = useRef(null);
  const vpProcessorRef = useRef(null);
  const vpChunksRef = useRef([]);
  const vpInputRateRef = useRef(VP_SAMPLE_RATE);
  const vpTimerRef = useRef(null);

  // Section refs for scrollspy
  const llmRef = useRef(null);
  const embeddingRef = useRef(null);
  const voiceprintRef = useRef(null);
  const trainingRef = useRef(null);
  const accountRef = useRef(null);
  const migrationRef = useRef(null);
  const sectionRefs = {
    llm: llmRef,
    embedding: embeddingRef,
    voiceprint: voiceprintRef,
    training: trainingRef,
    account: accountRef,
    migration: migrationRef,
  };
  const scrollSpyLock = useRef(0);

  // 数据迁移状态
  const [exporting, setExporting] = useState(false);
  const [importFile, setImportFile] = useState(null);
  const [importDbStrategy, setImportDbStrategy] = useState("skip");
  const [importOverwriteFiles, setImportOverwriteFiles] = useState(false);
  const [importConfirming, setImportConfirming] = useState(false);
  const [importBusy, setImportBusy] = useState(false);
  const [migrationMessage, setMigrationMessage] = useState("");
  const [migrationError, setMigrationError] = useState("");
  const importFileInputRef = useRef(null);

  useEffect(() => {
    getSettings()
      .then((data) => {
        setApiBase(data.llm.api_base || "");
        setApiKey(data.llm.api_key || "");
        setModel(data.llm.model || "");
        setTemperature(data.llm.temperature ?? 0.7);
        const emb = data.embedding || {};
        setEmbBackend(emb.backend || "");
        setEmbApiBase(emb.api_base || "");
        setEmbApiKey(emb.api_key || "");
        setEmbApiModel(emb.api_model || "");
        setEmbLocalModel(emb.local_model || "");
        setEmbLocalPath(emb.local_path || "");
        setAllowRegistration(Boolean(data.system?.allow_registration));
        setNumQuestions(data.training.num_questions ?? 10);
        setDivergence(data.training.divergence ?? 3);
      })
      .catch((err) => setError("加载设置失败: " + err.message))
      .finally(() => setLoading(false));

    getVoiceprintStatus()
      .then((s) => setVpStatus(s))
      .catch(() => {});
  }, []);

  const cleanupRecorder = useCallback(() => {
    if (vpTimerRef.current != null) {
      clearInterval(vpTimerRef.current);
      vpTimerRef.current = null;
    }
    vpProcessorRef.current?.disconnect();
    vpProcessorRef.current = null;
    vpSourceRef.current?.disconnect();
    vpSourceRef.current = null;
    vpStreamRef.current?.getTracks().forEach((t) => t.stop());
    vpStreamRef.current = null;
    vpCtxRef.current?.close().catch(() => {});
    vpCtxRef.current = null;
    setVpRecording(false);
    setVpRecordingSec(0);
  }, []);

  useEffect(() => () => cleanupRecorder(), [cleanupRecorder]);

  // ScrollSpy: highlight tab whose section is most prominent in the viewport
  useEffect(() => {
    if (loading) return;
    const anyEl = llmRef.current;
    if (!anyEl) return;
    const scroller = anyEl.closest("main") || null;

    const observer = new IntersectionObserver(
      (entries) => {
        if (Date.now() < scrollSpyLock.current) return;
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible.length === 0) return;
        const id = visible[0].target.getAttribute("data-tab-id");
        if (id) setActiveTab(id);
      },
      {
        root: scroller,
        rootMargin: "-15% 0px -55% 0px",
        threshold: 0,
      }
    );

    Object.values(sectionRefs).forEach((ref) => {
      if (ref.current) observer.observe(ref.current);
    });

    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading]);

  const handleTabClick = (id) => {
    setActiveTab(id);
    const el = sectionRefs[id]?.current;
    if (!el) return;
    // Suppress scrollspy briefly while the smooth scroll plays out
    scrollSpyLock.current = Date.now() + 700;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const handleSaveVpCredentials = async () => {
    setVpBusy(true);
    setVpMessage("");
    try {
      await putVoiceprintCredentials({
        secret_id: vpSecretId.trim(),
        secret_key: vpSecretKey.trim(),
        app_id: vpAppId.trim(),
      });
      const s = await getVoiceprintStatus();
      setVpStatus(s);
      setVpMessage("凭据已验证并保存");
    } catch (err) {
      setVpMessage("保存失败：" + (err.message || "未知错误"));
    } finally {
      setVpBusy(false);
    }
  };

  const startVpRecording = async () => {
    setVpMessage("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: VP_SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      const ctx = new AudioContext({ sampleRate: VP_SAMPLE_RATE });
      vpInputRateRef.current = ctx.sampleRate;
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      vpChunksRef.current = [];

      processor.onaudioprocess = (e) => {
        const ch = e.inputBuffer.getChannelData(0);
        vpChunksRef.current.push(new Float32Array(ch));
      };
      source.connect(processor);
      processor.connect(ctx.destination);

      vpStreamRef.current = stream;
      vpCtxRef.current = ctx;
      vpSourceRef.current = source;
      vpProcessorRef.current = processor;

      setVpRecording(true);
      setVpRecordingSec(0);
      const t0 = Date.now();
      vpTimerRef.current = setInterval(() => {
        setVpRecordingSec((Date.now() - t0) / 1000);
      }, 200);
    } catch (err) {
      cleanupRecorder();
      setVpMessage("麦克风访问失败：" + (err.message || "未知错误"));
    }
  };

  const stopVpRecording = async () => {
    const chunks = vpChunksRef.current;
    const inputRate = vpInputRateRef.current;
    const seconds = vpRecordingSec;
    cleanupRecorder();

    if (seconds < VP_MIN_SECONDS) {
      setVpMessage(`录音太短，至少 ${VP_MIN_SECONDS} 秒`);
      return;
    }

    setVpBusy(true);
    try {
      const merged = mergeFloat32(chunks);
      const pcm = resampleToPcm16(merged, inputRate, VP_SAMPLE_RATE);
      const wav = encodeWav(pcm, VP_SAMPLE_RATE);
      await enrollVoiceprint(wav);
      const s = await getVoiceprintStatus();
      setVpStatus(s);
      setVpMessage("声纹已注册");
    } catch (err) {
      setVpMessage("注册失败：" + (err.message || "未知错误"));
    } finally {
      setVpBusy(false);
    }
  };

  const handleDeleteEnrollment = async () => {
    setVpBusy(true);
    setVpMessage("");
    try {
      await deleteVoiceprintEnrollment();
      const s = await getVoiceprintStatus();
      setVpStatus(s);
      setVpMessage("已删除已注册声纹");
    } catch (err) {
      setVpMessage("删除失败：" + (err.message || "未知错误"));
    } finally {
      setVpBusy(false);
    }
  };

  const handleExport = async () => {
    setExporting(true);
    setMigrationError("");
    setMigrationMessage("");
    try {
      const { filename, size } = await exportData();
      const sizeMb = (size / 1024 / 1024).toFixed(2);
      setMigrationMessage(`已下载 ${filename} (${sizeMb} MB)`);
    } catch (err) {
      setMigrationError("导出失败：" + (err.message || "未知错误"));
    } finally {
      setExporting(false);
    }
  };

  const handleImportFileChange = (e) => {
    const f = e.target.files?.[0] || null;
    setImportFile(f);
    setImportConfirming(false);
    setMigrationMessage("");
    setMigrationError("");
  };

  const handleImportClick = () => {
    if (!importFile) return;
    setImportConfirming(true);
  };

  const handleImportConfirm = async () => {
    if (!importFile) return;
    setImportBusy(true);
    setMigrationError("");
    setMigrationMessage("");
    try {
      const r = await importData(importFile, {
        dbStrategy: importDbStrategy,
        overwriteFiles: importOverwriteFiles,
      });
      setMigrationMessage(
        `已导入：会话写入/更新 ${r.db_inserted} 条，跳过 ${r.db_skipped} 条；文件复制 ${r.files_copied} 个，跳过 ${r.files_skipped} 个。建议刷新页面以加载新数据。`
      );
      setImportFile(null);
      setImportConfirming(false);
      if (importFileInputRef.current) importFileInputRef.current.value = "";
    } catch (err) {
      setMigrationError("导入失败：" + (err.message || "未知错误"));
    } finally {
      setImportBusy(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError("");
    try {
      await updateSettings({
        llm: { api_base: apiBase, api_key: apiKey, model, temperature },
        embedding: {
          backend: embBackend,
          api_base: embApiBase,
          api_key: embApiKey,
          api_model: embApiModel,
          local_model: embLocalModel,
          local_path: embLocalPath,
        },
        system: { allow_registration: allowRegistration },
        training: { num_questions: numQuestions, divergence },
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      setError("保存失败: " + err.message);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-dim">
        <Loader2 size={20} className="animate-spin" />
      </div>
    );
  }

  const labelClass = "text-[11px] font-semibold uppercase tracking-[0.18em] text-dim/80";
  const inputClass = "h-12 rounded-2xl bg-card/90";

  const TABS = [
    { id: "llm", label: "LLM 服务", icon: Server },
    { id: "embedding", label: "Embedding", icon: Boxes },
    { id: "voiceprint", label: "声纹识别", icon: Mic },
    { id: "training", label: "训练参数", icon: Sliders },
    { id: "account", label: "账户", icon: UserCog },
    { id: "migration", label: "数据迁移", icon: Database },
  ];

  return (
    <div className="flex-1 w-full max-w-[1080px] mx-auto px-4 pt-6 pb-0 md:px-7 md:pt-8">
      <div className="mb-7">
        <div className="text-2xl md:text-[28px] font-display font-bold">设置</div>
        <div className="text-sm text-dim mt-1">配置 LLM 服务和训练参数</div>
      </div>

      <div className="grid gap-5 lg:grid-cols-[220px_1fr]">
        {/* Left Tab Rail */}
        <nav className="lg:sticky lg:top-4 lg:self-start">
          <div className="flex gap-1 overflow-x-auto lg:flex-col lg:gap-0.5 lg:overflow-visible">
            {TABS.map((tab) => {
              const { id, label } = tab;
              const Icon = tab.icon;
              const active = activeTab === id;
              return (
                <button
                  key={id}
                  onClick={() => handleTabClick(id)}
                  className={cn(
                    "group relative flex items-center gap-2.5 rounded-xl px-3 py-2 text-left text-[13px] transition-all duration-300 shrink-0 lg:w-full",
                    active
                      ? "bg-primary/10 text-primary font-medium"
                      : "text-dim hover:text-text hover:bg-hover"
                  )}
                >
                  {active && (
                    <div className="absolute left-0 top-1/2 hidden h-5 w-[3px] -translate-y-1/2 rounded-r-full bg-primary drop-shadow-[0_0_4px_currentColor] lg:block" />
                  )}
                  <Icon
                    size={16}
                    className={cn("shrink-0", active ? "text-primary" : "text-dim group-hover:text-primary")}
                  />
                  <span className="truncate">{label}</span>
                </button>
              );
            })}
          </div>
        </nav>

        {/* Right Content Pane */}
        <div className="min-w-0 space-y-5">
        {/* LLM Provider */}
        <Card ref={llmRef} data-tab-id="llm" className="overflow-hidden border-border/40 bg-card/40 scroll-mt-4">
          <CardContent className="p-5 md:p-7">
            <div className="flex items-center gap-2 mb-1">
              <Server size={16} className="text-primary" />
              <span className="text-base font-semibold">LLM 服务配置</span>
            </div>
            <div className="text-[13px] text-dim mb-6">更改后立即生效，无需重启后端</div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label className={labelClass}>API Base URL</Label>
                <Input
                  className={inputClass}
                  placeholder="例：https://api.openai.com/v1"
                  value={apiBase}
                  onChange={(e) => setApiBase(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label className={labelClass}>Model</Label>
                <Input
                  className={inputClass}
                  placeholder="例：gpt-4o"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                />
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 mt-4">
              <div className="space-y-2">
                <Label className={labelClass}>API Key</Label>
                <div className="relative">
                  <Input
                    className={cn(inputClass, "pr-11")}
                    type={showKey ? "text" : "password"}
                    placeholder="sk-..."
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                  />
                  <button
                    type="button"
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-dim hover:text-text transition-colors"
                    onClick={() => setShowKey((v) => !v)}
                  >
                    {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>
              <div className="space-y-2">
                <Label className={labelClass}>Temperature</Label>
                <Input
                  className={inputClass}
                  type="number"
                  step={0.1}
                  min={0}
                  max={2}
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value) || 0)}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Embedding */}
        <Card ref={embeddingRef} data-tab-id="embedding" className="overflow-hidden border-border/40 bg-card/40 scroll-mt-4">
          <CardContent className="p-5 md:p-7">
            <div className="flex items-center gap-2 mb-1">
              <Boxes size={16} className="text-primary" />
              <span className="text-base font-semibold">Embedding 模型</span>
            </div>
            <div className="text-[13px] text-dim mb-6">
              用于题库 / 简历 / 知识库的向量化。改动保存后立即热切换，已建索引不会重建。
            </div>

            <div className="space-y-2.5 mb-5">
              <Label className={labelClass}>后端模式</Label>
              <div className="flex flex-wrap gap-2">
                {[
                  { value: "", label: "自动", hint: "填了 API 走 API，否则本地" },
                  { value: "api", label: "API", hint: "OpenAI 兼容接口" },
                  { value: "local", label: "本地", hint: "HuggingFace 模型" },
                ].map((opt) => (
                  <button
                    key={opt.value || "auto"}
                    type="button"
                    onClick={() => setEmbBackend(opt.value)}
                    className={cn(
                      "px-4 py-2 rounded-xl border text-sm transition-all",
                      embBackend === opt.value
                        ? "bg-primary/12 text-primary border-primary/50 font-medium"
                        : "border-border bg-card/80 text-dim hover:text-text hover:bg-hover"
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              <div className="text-[12px] text-dim/70 mt-1 min-h-[18px]">
                {[
                  { value: "", hint: "填了 API 字段走 API，否则走本地（兼容老配置）" },
                  { value: "api", hint: "通过 OpenAI 兼容接口请求 embedding" },
                  { value: "local", hint: "用 HuggingFace 加载本地模型，需 `pip install -r requirements.local-embedding.txt`" },
                ].find((o) => o.value === embBackend)?.hint}
              </div>
            </div>

            {(embBackend === "" || embBackend === "api") && (
              <div className="space-y-4">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-dim/60">API 模式</div>
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label className={labelClass}>API Base URL</Label>
                    <Input
                      className={inputClass}
                      placeholder="例：https://api.openai.com/v1（OpenAI 官方留空亦可）"
                      value={embApiBase}
                      onChange={(e) => setEmbApiBase(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className={labelClass}>Embedding Model</Label>
                    <Input
                      className={inputClass}
                      placeholder="例：BAAI/bge-m3"
                      value={embApiModel}
                      onChange={(e) => setEmbApiModel(e.target.value)}
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className={labelClass}>API Key</Label>
                  <div className="relative">
                    <Input
                      className={cn(inputClass, "pr-11")}
                      type={showEmbKey ? "text" : "password"}
                      placeholder="sk-..."
                      value={embApiKey}
                      onChange={(e) => setEmbApiKey(e.target.value)}
                    />
                    <button
                      type="button"
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-dim hover:text-text transition-colors"
                      onClick={() => setShowEmbKey((v) => !v)}
                    >
                      {showEmbKey ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {(embBackend === "" || embBackend === "local") && (
              <div className={cn("space-y-4", embBackend === "" && "mt-6 border-t border-border/40 pt-5")}>
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-dim/60">本地模式</div>
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label className={labelClass}>Model Name</Label>
                    <Input
                      className={inputClass}
                      placeholder="例：BAAI/bge-m3"
                      value={embLocalModel}
                      onChange={(e) => setEmbLocalModel(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className={labelClass}>本地路径 (可选)</Label>
                    <Input
                      className={inputClass}
                      placeholder="留空时按 model name 在线下载"
                      value={embLocalPath}
                      onChange={(e) => setEmbLocalPath(e.target.value)}
                    />
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Voiceprint (Optional) */}
        <Card ref={voiceprintRef} data-tab-id="voiceprint" className="overflow-hidden border-border/40 bg-card/40 scroll-mt-4">
          <CardContent className="p-5 md:p-7">
            <div className="flex items-center gap-2 mb-1">
              <Mic size={16} className="text-primary" />
              <span className="text-base font-semibold">声纹识别（可选）</span>
            </div>
            <div className="text-[13px] text-dim mb-5">
              配置腾讯云 VPR 凭据并提前录入候选人声纹后，实时面试中自动识别 HR 与候选人，无需手动切换。未配置时保持手动按钮模式。
            </div>

            <div className="rounded-xl border border-border/60 bg-background/40 px-4 py-3 mb-5">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-dim/80 mb-1">
                当前状态
              </div>
              <div className="text-sm">
                {vpStatus.enrolled ? (
                  <span className="text-primary">● 已注册 {vpStatus.enrolled_at ? `(${vpStatus.enrolled_at.slice(0, 10)})` : ""}</span>
                ) : vpStatus.configured ? (
                  <span className="text-dim">◐ 已配置凭据，尚未注册声纹</span>
                ) : (
                  <span className="text-dim/70">○ 未配置</span>
                )}
              </div>
            </div>

            <div className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label className={labelClass}>Secret Id</Label>
                  <Input className={inputClass} value={vpSecretId} onChange={(e) => setVpSecretId(e.target.value)} placeholder="AKID..." />
                </div>
                <div className="space-y-2">
                  <Label className={labelClass}>App Id (可选)</Label>
                  <Input className={inputClass} value={vpAppId} onChange={(e) => setVpAppId(e.target.value)} placeholder="留空即可" />
                </div>
              </div>

              <div className="space-y-2">
                <Label className={labelClass}>Secret Key</Label>
                <div className="relative">
                  <Input
                    className={cn(inputClass, "pr-11")}
                    type={showVpKey ? "text" : "password"}
                    value={vpSecretKey}
                    onChange={(e) => setVpSecretKey(e.target.value)}
                    placeholder="腾讯云 Secret Key"
                  />
                  <button
                    type="button"
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-dim hover:text-text transition-colors"
                    onClick={() => setShowVpKey((v) => !v)}
                  >
                    {showVpKey ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-3 pt-1">
                <Button
                  variant="outline"
                  disabled={vpBusy || vpRecording || !vpSecretId || !vpSecretKey}
                  onClick={handleSaveVpCredentials}
                >
                  测试并保存凭据
                </Button>
              </div>

              <div className="border-t border-border/40 pt-5 mt-2">
                <Label className={labelClass}>候选人声纹</Label>
                <div className="text-[12px] text-dim/70 mt-1 mb-3">
                  {vpRecording
                    ? `录音中：${vpRecordingSec.toFixed(1)} 秒`
                    : `建议连续说话 ≥ ${VP_MIN_SECONDS} 秒，单人、安静环境`}
                </div>
                <div className="flex flex-wrap gap-3">
                  {vpRecording ? (
                    <Button
                      variant="outline"
                      disabled={vpBusy}
                      onClick={stopVpRecording}
                      className="border-red-400/50 text-red-500 hover:bg-red-500/10"
                    >
                      <Square size={14} className="mr-1.5" />
                      结束并上传
                    </Button>
                  ) : (
                    <Button
                      variant="outline"
                      disabled={vpBusy || !vpStatus.configured}
                      onClick={startVpRecording}
                    >
                      <Mic size={14} className="mr-1.5" />
                      {vpStatus.enrolled ? "重新录制" : "开始录制"}
                    </Button>
                  )}
                  {vpStatus.enrolled && !vpRecording && (
                    <Button
                      variant="outline"
                      disabled={vpBusy}
                      onClick={handleDeleteEnrollment}
                      className="border-border/60 hover:border-red-400/50 hover:text-red-500"
                    >
                      <Trash2 size={14} className="mr-1.5" />
                      删除声纹
                    </Button>
                  )}
                </div>
              </div>

              {vpMessage && (
                <div className="text-[12px] text-dim pt-1">{vpMessage}</div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Training Params */}
        <Card ref={trainingRef} data-tab-id="training" className="overflow-hidden border-border/40 bg-card/40 scroll-mt-4">
          <CardContent className="p-5 md:p-7">
            <div className="flex items-center gap-2 mb-1">
              <Sliders size={16} className="text-primary" />
              <span className="text-base font-semibold">训练参数</span>
            </div>
            <div className="text-[13px] text-dim mb-6">每次开始专项训练时的默认设置</div>

            <div className="space-y-5">
              <div className="space-y-2">
                <Label className={labelClass}>每轮题目数</Label>
                <Input
                  className={cn(inputClass, "max-w-[140px]")}
                  type="number"
                  min={5}
                  max={20}
                  value={numQuestions}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (v >= 5 && v <= 20) setNumQuestions(v);
                    else if (e.target.value === "") setNumQuestions(5);
                  }}
                />
                <div className="text-[12px] text-dim/60">范围 5 – 20，默认 10</div>
              </div>

              <div className="space-y-2.5">
                <Label className={labelClass}>题目发散度</Label>
                <div className="flex flex-wrap gap-2">
                  {DIVERGENCE_OPTIONS.map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setDivergence(opt.value)}
                      className={cn(
                        "px-4 py-2 rounded-xl border text-sm transition-all",
                        divergence === opt.value
                          ? "bg-primary/12 text-primary border-primary/50 font-medium"
                          : "border-border bg-card/80 text-dim hover:text-text hover:bg-hover"
                      )}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
                <div className="text-[12px] text-dim/70 mt-1 min-h-[18px]">
                  {DIVERGENCE_OPTIONS.find((o) => o.value === divergence)?.description}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Account / System */}
        <Card ref={accountRef} data-tab-id="account" className="overflow-hidden border-border/40 bg-card/40 scroll-mt-4">
          <CardContent className="p-5 md:p-7">
            <div className="flex items-center gap-2 mb-1">
              <UserCog size={16} className="text-primary" />
              <span className="text-base font-semibold">账户</span>
            </div>
            <div className="text-[13px] text-dim mb-5">控制谁能进入系统。保存后立即生效。</div>

            <label className="flex items-start justify-between gap-4 rounded-xl border border-border/60 bg-background/40 px-4 py-4 cursor-pointer select-none">
              <div className="min-w-0">
                <div className="text-sm font-medium">允许新用户注册</div>
                <div className="text-[12px] text-dim/70 mt-1 leading-5">
                  关闭后登录页隐藏注册入口，只有 DEFAULT_EMAIL/PASSWORD（或已注册账户）能登录。建议自用部署关闭。
                </div>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={allowRegistration}
                onClick={() => setAllowRegistration((v) => !v)}
                className={cn(
                  "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors duration-200 mt-0.5",
                  allowRegistration ? "bg-primary" : "bg-border"
                )}
              >
                <span
                  className={cn(
                    "inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform duration-200",
                    allowRegistration ? "translate-x-5" : "translate-x-0.5"
                  )}
                />
              </button>
            </label>
          </CardContent>
        </Card>

        {/* Data Migration */}
        <Card ref={migrationRef} data-tab-id="migration" className="overflow-hidden border-border/40 bg-card/40 scroll-mt-4">
          <CardContent className="p-5 md:p-7">
            <div className="flex items-center gap-2 mb-1">
              <Database size={16} className="text-primary" />
              <span className="text-base font-semibold">数据迁移</span>
            </div>
            <div className="text-[13px] text-dim mb-5">
              备份与跨机器迁移面试历史、复盘、画像、简历、知识库、题库与训练偏好。LLM/声纹凭据等环境配置不在此范围。
            </div>

            <div className="space-y-5">
              {/* Export */}
              <div className="rounded-xl border border-border/60 bg-background/40 px-4 py-4">
                <div className="flex items-start justify-between gap-3 flex-wrap">
                  <div>
                    <div className="text-sm font-medium mb-0.5">导出当前账户数据</div>
                    <div className="text-[12px] text-dim/70">打包为 .tar.gz 下载到本地</div>
                  </div>
                  <Button variant="outline" disabled={exporting} onClick={handleExport}>
                    {exporting ? (
                      <Loader2 size={14} className="mr-1.5 animate-spin" />
                    ) : (
                      <Download size={14} className="mr-1.5" />
                    )}
                    {exporting ? "导出中..." : "导出"}
                  </Button>
                </div>
              </div>

              {/* Import */}
              <div className="rounded-xl border border-border/60 bg-background/40 px-4 py-4 space-y-4">
                <div>
                  <div className="text-sm font-medium mb-0.5">从备份导入</div>
                  <div className="text-[12px] text-dim/70">归档中的数据将归到当前登录账户</div>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <input
                    ref={importFileInputRef}
                    type="file"
                    accept=".gz,.tgz,application/gzip,application/x-gzip"
                    onChange={handleImportFileChange}
                    className="text-[12px] text-dim file:mr-3 file:rounded-lg file:border-0 file:bg-card file:px-3 file:py-1.5 file:text-sm file:text-text hover:file:bg-hover"
                  />
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label className={labelClass}>会话冲突策略</Label>
                    <div className="flex gap-2">
                      {[
                        { value: "skip", label: "保留本地" },
                        { value: "overwrite", label: "用归档覆盖" },
                      ].map((opt) => (
                        <button
                          key={opt.value}
                          type="button"
                          onClick={() => setImportDbStrategy(opt.value)}
                          className={cn(
                            "px-3 py-1.5 rounded-lg border text-[13px] transition-all",
                            importDbStrategy === opt.value
                              ? "bg-primary/12 text-primary border-primary/50 font-medium"
                              : "border-border bg-card/80 text-dim hover:text-text hover:bg-hover"
                          )}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label className={labelClass}>文件冲突</Label>
                    <label className="flex items-center gap-2 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={importOverwriteFiles}
                        onChange={(e) => setImportOverwriteFiles(e.target.checked)}
                        className="accent-primary"
                      />
                      <span className="text-[13px] text-dim">用归档文件覆盖本地</span>
                    </label>
                  </div>
                </div>

                {importConfirming ? (
                  <div className="rounded-lg border border-amber-400/40 bg-amber-400/8 px-3 py-3">
                    <div className="flex items-start gap-2 mb-2.5">
                      <AlertTriangle size={14} className="text-amber-500 mt-0.5 shrink-0" />
                      <div className="text-[13px]">
                        将把 <span className="font-medium">{importFile?.name}</span> 合并到当前账户。
                        {importDbStrategy === "overwrite" && "本地同 ID 的会话会被覆盖。"}
                        {importOverwriteFiles && "用户文件也会被覆盖。"}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button variant="outline" disabled={importBusy} onClick={() => setImportConfirming(false)}>
                        取消
                      </Button>
                      <Button variant="gradient" disabled={importBusy} onClick={handleImportConfirm}>
                        {importBusy && <Loader2 size={14} className="mr-1.5 animate-spin" />}
                        {importBusy ? "导入中..." : "确认导入"}
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div>
                    <Button variant="outline" disabled={!importFile || importBusy} onClick={handleImportClick}>
                      <Upload size={14} className="mr-1.5" />
                      导入
                    </Button>
                  </div>
                )}
              </div>

              {(migrationMessage || migrationError) && (
                <div className={cn("text-[12px]", migrationError ? "text-red" : "text-dim")}>
                  {migrationError || migrationMessage}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        </div>
      </div>

      {/* Sticky save bar (commits LLM + training params; 声纹/数据迁移 各自保存) */}
      <div className="sticky bottom-0 z-10 -mx-4 mt-6 border-t border-border/40 bg-background/85 px-4 py-3 backdrop-blur-md md:-mx-7 md:px-7">
        <div className="flex items-center justify-end gap-4">
          {error ? (
            <span className="text-sm text-red">{error}</span>
          ) : (
            <span className="text-[12px] text-dim/70">
              保存 LLM + Embedding + 训练参数 + 账户。声纹与数据迁移各自独立保存。
            </span>
          )}
          <Button variant="gradient" className="px-8" onClick={handleSave} disabled={saving}>
            {saving ? <Loader2 size={15} className="animate-spin" /> : saved ? <Check size={15} /> : null}
            {saving ? "保存中..." : saved ? "已保存" : "保存"}
          </Button>
        </div>
      </div>
    </div>
  );
}
