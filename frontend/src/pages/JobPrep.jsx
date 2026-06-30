import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  BriefcaseBusiness,
  CheckCircle2,
  FileText,
  Loader2,
  ShieldAlert,
  Sparkles,
  Target,
} from "lucide-react";
import { getResumeStatus, previewJobPrep, startJobPrep } from "../api/interview";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const PAGE_CLASS = "flex-1 w-full max-w-[1600px] mx-auto px-4 py-6 md:px-7 md:py-8 xl:px-10 2xl:px-12";

// Survive leaving the page without starting practice — a JD analysis costs an LLM
// call, so persist inputs + result locally and restore them on return.
const DRAFT_KEY = "jobprep-draft";

function loadDraft() {
  try {
    return JSON.parse(localStorage.getItem(DRAFT_KEY)) || {};
  } catch {
    return {};
  }
}

function priorityVariant(priority) {
  if (priority === "high") return "destructive";
  if (priority === "medium") return "blue";
  return "secondary";
}

function formatFileSize(size) {
  if (!size) return null;
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function buildStatus({ preview, previewStale, previewing, starting }) {
  if (starting) return { label: "初始化中", tone: "blue", hint: "正在创建定向训练" };
  if (previewing) return { label: "分析中", tone: "blue", hint: "正在拆解岗位重点" };
  if (preview && previewStale) return { label: "待更新", tone: "amber", hint: "岗位信息已变更" };
  if (preview) return { label: "可开练", tone: "green", hint: "分析结果已就绪" };
  return { label: "待分析", tone: "neutral", hint: "先生成岗位拆解" };
}

function toneClasses(tone) {
  if (tone === "green") return "border-green/20 bg-green/8 text-green";
  if (tone === "blue") return "border-blue-500/20 bg-blue-500/8 text-blue-300";
  if (tone === "amber") return "border-amber-500/20 bg-amber-500/10 text-amber-300";
  return "border-border/80 bg-card/82 text-text";
}

export default function JobPrep() {
  const navigate = useNavigate();
  const initialDraft = useMemo(loadDraft, []);
  const [company, setCompany] = useState(initialDraft.company || "");
  const [position, setPosition] = useState(initialDraft.position || "");
  const [jdText, setJdText] = useState(initialDraft.jdText || "");
  const [resumeFile, setResumeFile] = useState(null);
  const [useResume, setUseResume] = useState(true);
  const [preview, setPreview] = useState(initialDraft.preview || null);
  const [previewSignature, setPreviewSignature] = useState(initialDraft.previewSignature || "");
  const [loadingResume, setLoadingResume] = useState(true);
  const [previewing, setPreviewing] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getResumeStatus()
      .then((data) => {
        if (data.has_resume) {
          setResumeFile({ filename: data.filename, size: data.size });
          setUseResume(true);
        } else {
          setUseResume(false);
        }
      })
      .catch(() => setUseResume(false))
      .finally(() => setLoadingResume(false));
  }, []);

  // Best-effort: keep the workspace mirrored to localStorage so a refresh or
  // navigation never discards a token-costing analysis. Cleared once practice starts.
  useEffect(() => {
    try {
      localStorage.setItem(DRAFT_KEY, JSON.stringify({ company, position, jdText, preview, previewSignature }));
    } catch {
      // storage full / unavailable — ignore
    }
  }, [company, position, jdText, preview, previewSignature]);

  const payload = useMemo(() => ({
    company: company.trim() || null,
    position: position.trim() || null,
    jd_text: jdText.trim(),
    use_resume: !!(useResume && resumeFile),
  }), [company, position, jdText, useResume, resumeFile]);

  const signature = JSON.stringify(payload);
  const charCount = payload.jd_text.length;
  const previewStale = !!preview && previewSignature !== signature;
  const canPreview = charCount >= 50 && !previewing && !starting;
  const canStart = !!preview && !previewStale && !previewing && !starting;
  const status = buildStatus({ preview, previewStale, previewing, starting });
  const resumeReady = !!resumeFile;
  const resumeEnabled = !!(useResume && resumeFile);
  const questionGroupCount = preview?.likely_question_groups?.length || 0;
  const focusCount = preview?.focus_areas?.length || 0;
  const priorityCount = preview?.prep_priorities?.length || 0;

  const handlePreview = async () => {
    setPreviewing(true);
    setError("");
    try {
      const data = await previewJobPrep(payload);
      setPreview(data.preview);
      setPreviewSignature(signature);
    } catch (err) {
      setError("JD 分析失败: " + err.message);
    } finally {
      setPreviewing(false);
    }
  };

  const handleStart = async () => {
    setStarting(true);
    setError("");
    try {
      const data = await startJobPrep({ ...payload, preview_data: preview });
      try { localStorage.removeItem(DRAFT_KEY); } catch { /* ignore */ }
      navigate(`/interview/${data.session_id}`, { state: data });
    } catch (err) {
      setError("启动失败: " + err.message);
      setStarting(false);
    }
  };

  return (
    <div className={PAGE_CLASS}>
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.55fr)_380px] 2xl:grid-cols-[minmax(0,1.65fr)_400px]">
        <div className="space-y-5">
          <Card className="overflow-hidden border-border/80 bg-card/76">
            <CardContent className="p-5 md:p-6 xl:p-7">
              <div className="flex flex-col gap-6">
                <div className="border-b border-border/70 pb-4">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-dim/80">岗位拆解工作区</div>
                  <div className="mt-2 text-2xl font-display font-bold tracking-tight md:text-3xl">JD 定向备面</div>
                  <div className="mt-1.5 max-w-2xl text-sm leading-6 text-dim">
                    先填岗位信息，再分析这个岗位真正会盯什么，以及你该不该立刻开始训练。
                  </div>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label className="text-[11px] font-semibold uppercase tracking-[0.18em] text-dim/80">公司</Label>
                    <Input
                      className="h-12 rounded-2xl bg-card/90"
                      placeholder="例：字节跳动"
                      value={company}
                      onChange={(event) => setCompany(event.target.value)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label className="text-[11px] font-semibold uppercase tracking-[0.18em] text-dim/80">岗位</Label>
                    <Input
                      className="h-12 rounded-2xl bg-card/90"
                      placeholder="例：AI 后台开发实习生"
                      value={position}
                      onChange={(event) => setPosition(event.target.value)}
                    />
                  </div>
                </div>

                <div className="rounded-[28px] border border-border/80 bg-background/65 p-4 md:p-5">
                  <div className="flex flex-col gap-3 border-b border-border/70 pb-4 md:flex-row md:items-end md:justify-between">
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-dim/80">岗位 JD</div>
                      <div className="mt-1 text-sm text-dim">
                        直接贴完整职责、要求、加分项。这里越完整，后面的岗位拆解越有价值。
                      </div>
                    </div>
                    <div className="rounded-full border border-border/80 bg-card/92 px-3 py-1 text-sm tabular-nums text-dim">
                      {charCount} 字
                    </div>
                  </div>

                  <Textarea
                    className="mt-4 min-h-[360px] rounded-[24px] border-border/70 bg-background/80 px-4 py-4 text-[15px] leading-7 resize-y md:min-h-[440px]"
                    placeholder="粘贴完整 JD。优先保留职责、任职要求、加分项、业务背景和技术栈。"
                    value={jdText}
                    onChange={(event) => setJdText(event.target.value)}
                  />

                  <div className="mt-4 grid gap-3 md:grid-cols-3">
                    <HintChip title="至少 50 字" description="低于这个长度，不值得分析。" />
                    <HintChip title="不要只贴标题" description="只写岗位名，基本拆不出重点。" />
                    <HintChip title="保留原始措辞" description="岗位关键词会影响追问方向。" />
                  </div>
                </div>

                <Card className="border-border/80 bg-[linear-gradient(135deg,rgba(255,255,255,0.94),rgba(244,247,255,0.92))] dark:bg-[linear-gradient(135deg,rgba(24,24,27,0.96),rgba(30,41,59,0.72))]">
                  <CardContent className="p-4 md:p-5">
                    <label className={cn("flex items-start gap-3", !resumeReady && "opacity-75")}>
                      <input
                        type="checkbox"
                        className="mt-1"
                        checked={resumeEnabled}
                        disabled={!resumeReady}
                        onChange={(event) => setUseResume(event.target.checked)}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <div className="text-sm font-semibold">结合简历生成针对性追问</div>
                          <Badge variant={resumeReady ? "blue" : "secondary"}>
                            {loadingResume ? "检查中" : resumeReady ? "已可用" : "未上传简历"}
                          </Badge>
                          {resumeFile?.size && (
                            <Badge variant="outline">
                              {formatFileSize(resumeFile.size)}
                            </Badge>
                          )}
                        </div>
                        <div className="mt-2 text-[13px] leading-6 text-dim">
                          {resumeReady
                            ? `已检测到简历：${resumeFile.filename}。开启后，会优先对照你的项目、经历和岗位要求来设计追问。`
                            : "当前没有可用简历。这不影响岗位拆解，但会少掉“岗位要求和你经历是否对位”这一层判断。"}
                        </div>
                      </div>
                    </label>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>

          {error && (
            <div className="rounded-2xl border border-red/20 bg-red/10 px-4 py-3 text-sm text-red">
              {error}
            </div>
          )}

          {previewStale && (
            <div className="rounded-2xl border border-amber-500/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
              你修改了 JD 或岗位信息，当前分析已经过时。重新分析后再开始训练。
            </div>
          )}
        </div>

        <div className="space-y-5 xl:sticky xl:top-6 xl:self-start">
          <Card className="overflow-hidden border-primary/15 bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.1),transparent_38%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(244,247,255,0.92))] dark:bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.12),transparent_34%),linear-gradient(180deg,rgba(24,24,27,0.98),rgba(30,41,59,0.84))]">
            <CardContent className="p-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-dim/80">决策面板</div>
                  <div className="mt-1 text-lg font-semibold">先判断值不值得练</div>
                </div>
                <div className={cn("rounded-full border px-3 py-1 text-sm", toneClasses(status.tone))}>
                  {status.label}
                </div>
              </div>

              <div className="mt-4 space-y-3">
                <StepRow
                  index="01"
                  title="整理岗位信息"
                  description={charCount >= 50 ? "JD 内容已经够用。" : "先把 JD 补到至少 50 字。"}
                  done={charCount >= 50}
                />
                <StepRow
                  index="02"
                  title="生成岗位拆解"
                  description={preview ? "已拿到考察点、补强项和提问方向。" : "分析后才知道这个岗位真正盯什么。"}
                  done={!!preview}
                  active={!preview}
                />
                <StepRow
                  index="03"
                  title="开始定向训练"
                  description={canStart ? "分析有效，可以直接进入训练。" : "只有分析结果有效时才建议开练。"}
                  done={canStart}
                  active={!!preview && !canStart}
                />
              </div>

              <div className="mt-5 grid grid-cols-2 gap-2">
                <MiniMetric label="JD 长度" value={charCount} />
                <MiniMetric label="简历联动" value={resumeEnabled ? "On" : "Off"} />
                <MiniMetric label="考察点" value={focusCount} />
                <MiniMetric label="提问组" value={questionGroupCount} />
              </div>

              <div className="mt-5 space-y-3">
                <Button
                  variant="gradient"
                  size="lg"
                  className="w-full"
                  disabled={!canPreview}
                  onClick={handlePreview}
                >
                  {previewing ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      分析中...
                    </>
                  ) : (
                    <>
                      <Sparkles size={18} />
                      先分析这个岗位
                    </>
                  )}
                </Button>

                <Button
                  variant={canStart ? "gradient" : "outline"}
                  size="lg"
                  className="w-full"
                  disabled={!canStart}
                  onClick={handleStart}
                >
                  {starting ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      初始化中...
                    </>
                  ) : (
                    "开始定向训练"
                  )}
                </Button>

                <Button variant="ghost" className="w-full" onClick={() => navigate("/")}>
                  返回首页
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="border-border/80">
            <CardContent className="p-5">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-dim/80">当前输入</div>
              <div className="mt-3 space-y-3 text-sm">
                <InfoRow label="公司" value={company.trim() || "未填写"} />
                <InfoRow label="岗位" value={position.trim() || "未填写"} />
                <InfoRow label="简历" value={resumeReady ? resumeFile.filename : "未检测到可用简历"} />
                <InfoRow label="模式" value={resumeEnabled ? "JD + 简历联动" : "仅 JD 分析"} />
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {preview ? (
        <div className="mt-6 space-y-5">
          <Card className="overflow-hidden border-primary/15 bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.08),transparent_30%),linear-gradient(135deg,rgba(255,255,255,0.98),rgba(242,246,255,0.92))] dark:bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.12),transparent_30%),linear-gradient(135deg,rgba(24,24,27,0.98),rgba(30,41,59,0.84))]">
            <CardContent className="p-5 md:p-6 xl:p-7">
              <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <BriefcaseBusiness size={18} className="text-blue-400" />
                    <div className="text-xl font-semibold">
                      {preview.company ? `${preview.company} · ` : ""}{preview.position || "目标岗位"}
                    </div>
                    <Badge variant={preview.resume_alignment?.resume_used ? "blue" : "secondary"}>
                      {preview.resume_alignment?.resume_used ? "JD + 简历联动" : "仅 JD 分析"}
                    </Badge>
                  </div>
                  <div className="mt-3 max-w-4xl text-sm leading-7 text-dim">
                    {preview.role_summary}
                  </div>
                </div>

                <div className="grid min-w-[240px] gap-2 sm:grid-cols-3 xl:grid-cols-1">
                  <ResultTag label="考察点" value={focusCount} />
                  <ResultTag label="补强项" value={priorityCount + (preview.resume_alignment?.risk_gaps?.length || 0)} />
                  <ResultTag label="提问组" value={questionGroupCount} />
                </div>
              </div>

              {preview.resume_alignment?.fit_assessment && (
                <div className="mt-5 rounded-2xl border border-blue-500/20 bg-blue-500/8 px-4 py-3 text-sm leading-7 text-blue-100">
                  <div className="mb-1 text-[13px] font-semibold text-blue-300">岗位匹配判断</div>
                  {preview.resume_alignment.fit_assessment}
                </div>
              )}
            </CardContent>
          </Card>

          <div className="grid gap-5 xl:grid-cols-2">
            <Card className="border-border/80">
              <CardContent className="p-5 md:p-6">
                <SectionTitle icon={<Target size={17} className="text-primary" />} title="核心考察点" />
                <div className="mt-4 space-y-3">
                  {(preview.focus_areas || []).map((item, index) => (
                    <div key={`${item.area}-${index}`} className="rounded-2xl border border-border/75 bg-card/75 px-4 py-3">
                      <div className="flex items-center justify-between gap-3">
                        <div className="text-sm font-semibold">{item.area}</div>
                        <Badge variant={priorityVariant(item.priority)}>{item.priority || "normal"}</Badge>
                      </div>
                      <div className="mt-2 text-[13px] leading-6 text-dim">{item.reason}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="border-border/80">
              <CardContent className="p-5 md:p-6">
                <SectionTitle icon={<ShieldAlert size={17} className="text-red" />} title="面试前优先补强" />
                <div className="mt-4 space-y-3">
                  {(preview.prep_priorities || []).map((item, index) => (
                    <div key={`${item}-${index}`} className="rounded-2xl border border-red/15 bg-red/8 px-4 py-3 text-sm leading-7">
                      {item}
                    </div>
                  ))}
                  {preview.resume_alignment?.risk_gaps?.map((item, index) => (
                    <div key={`gap-${index}`} className="rounded-2xl border border-border/70 bg-card/80 px-4 py-3 text-sm leading-7 text-dim">
                      {item}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {(preview.resume_alignment?.matching_evidence?.length > 0 || preview.resume_alignment?.recommended_stories?.length > 0) && (
            <Card className="border-border/80">
              <CardContent className="p-5 md:p-6">
                <SectionTitle icon={<FileText size={17} className="text-green" />} title="简历对位建议" />
                <div className="mt-4 grid gap-4 lg:grid-cols-2">
                  <div className="rounded-[24px] border border-green/15 bg-green/8 p-4">
                    <div className="text-[13px] font-semibold text-green">你现在能打的点</div>
                    <div className="mt-3 space-y-2">
                      {(preview.resume_alignment?.matching_evidence || []).map((item, index) => (
                        <div key={`evidence-${index}`} className="rounded-2xl border border-green/15 bg-background/70 px-4 py-3 text-sm leading-7">
                          {item}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-[24px] border border-border/75 bg-card/75 p-4">
                    <div className="text-[13px] font-semibold text-primary">优先拿来讲的经历</div>
                    <div className="mt-3 space-y-2">
                      {(preview.resume_alignment?.recommended_stories || []).map((item, index) => (
                        <div key={`story-${index}`} className="rounded-2xl border border-border/70 bg-background/70 px-4 py-3">
                          <div className="text-sm font-semibold">{item.project}</div>
                          <div className="mt-1 text-[13px] leading-6 text-dim">{item.reason}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          <Card className="border-border/80">
            <CardContent className="p-5 md:p-6">
              <SectionTitle icon={<Sparkles size={17} className="text-primary" />} title="高概率提问方向" />
              <div className="mt-4 grid gap-4 lg:grid-cols-2">
                {(preview.likely_question_groups || []).map((group, index) => (
                  <div key={`${group.title}-${index}`} className="rounded-[24px] border border-border/75 bg-card/75 p-4">
                    <div className="text-sm font-semibold">{group.title}</div>
                    <div className="mt-2 text-[13px] leading-6 text-dim">{group.reason}</div>
                    <div className="mt-4 space-y-2">
                      {(group.sample_questions || []).map((question, questionIndex) => (
                        <div key={`${question}-${questionIndex}`} className="rounded-2xl bg-background/80 px-3.5 py-3 text-sm leading-7">
                          {question}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      ) : (
        <Card className="mt-6 border-dashed border-border/80 bg-card/55">
          <CardContent className="p-8 text-center">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <Sparkles size={20} />
            </div>
            <div className="mt-4 text-lg font-semibold">分析结果会在这里展开</div>
            <div className="mt-2 text-sm leading-6 text-dim">
              包括岗位核心考察点、优先补强项、简历对位判断，以及高概率追问方向。
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function HintChip({ title, description }) {
  return (
    <div className="rounded-2xl border border-border/70 bg-background/72 px-3.5 py-3">
      <div className="text-sm font-semibold">{title}</div>
      <div className="mt-1 text-[13px] leading-6 text-dim">{description}</div>
    </div>
  );
}

function StepRow({ index, title, description, done = false, active = false }) {
  return (
    <div className={cn("rounded-2xl border px-3.5 py-3", done ? "border-green/20 bg-green/8" : active ? "border-primary/25 bg-primary/6" : "border-border/75 bg-card/72")}>
      <div className="flex items-start gap-3">
        <div className={cn("mt-0.5 flex h-7 w-7 items-center justify-center rounded-full text-[11px] font-semibold", done ? "bg-green/15 text-green" : active ? "bg-primary/12 text-primary" : "bg-hover text-dim")}>
          {done ? <CheckCircle2 size={14} /> : index}
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold">{title}</div>
          <div className="mt-1 text-[13px] leading-6 text-dim">{description}</div>
        </div>
      </div>
    </div>
  );
}

function MiniMetric({ label, value }) {
  return (
    <div className="rounded-2xl border border-border/75 bg-card/75 px-3 py-2.5">
      <div className="text-[11px] uppercase tracking-[0.16em] text-dim/80">{label}</div>
      <div className="mt-1 text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function InfoRow({ label, value }) {
  return (
    <div className="flex items-start justify-between gap-3 rounded-2xl border border-border/70 bg-card/72 px-3.5 py-3">
      <div className="shrink-0 text-dim">{label}</div>
      <div className="min-w-0 text-right font-medium">{value}</div>
    </div>
  );
}

function ResultTag({ label, value }) {
  return (
    <div className="rounded-2xl border border-border/75 bg-card/78 px-3 py-2.5">
      <div className="text-[11px] uppercase tracking-[0.16em] text-dim/80">{label}</div>
      <div className="mt-1 text-lg font-semibold tabular-nums">{value}</div>
    </div>
  );
}

function SectionTitle({ icon, title }) {
  return (
    <div className="flex items-center gap-2">
      {icon}
      <div className="font-semibold">{title}</div>
    </div>
  );
}
