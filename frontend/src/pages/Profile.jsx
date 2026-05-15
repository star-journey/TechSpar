import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  Brain,
  Clock3,
  Target,
  TrendingUp,
} from "lucide-react";

import { getProfile, getTopics } from "../api/interview";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

import DomainTable from "./profile/DomainTable";
import EvidenceTable from "./profile/EvidenceTable";
import {
  BehaviorSignalList,
  CollapsibleSection,
  HabitTagList,
  PatternColumn,
  PerformanceDimCard,
  ScoreChart,
  SectionHeader,
  TopicPriorityCard,
} from "./profile/components";
import {
  buildBehaviorSignals,
  buildDomainInsights,
  buildModeCounts,
  buildPriorityWeaknesses,
  buildTrainingModeStats,
  formatMinute,
  formatShortDate,
  getLatestEntry,
  getRealTopicSet,
  getTrendDelta,
  isKnowledgeAxis,
  sortByDateDesc,
} from "./profile/derive";
import { MODE_META, PAGE_CLASS, PERFORMANCE_DIMENSIONS } from "./profile/meta";

export default function Profile() {
  const [profile, setProfile] = useState(null);
  const [canonicalTopics, setCanonicalTopics] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    Promise.all([
      getProfile().catch(() => null),
      getTopics().catch(() => ({})),
    ])
      .then(([nextProfile, topics]) => {
        setProfile(nextProfile);
        setCanonicalTopics(new Set(Object.keys(topics || {})));
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className={cn(PAGE_CLASS, "space-y-4")}>
        <Skeleton className="h-8 w-32" />
        <Skeleton className="h-5 w-72" />
        <Skeleton className="h-[220px] w-full rounded-[28px]" />
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(340px,0.75fr)]">
          <Skeleton className="h-[280px] rounded-[24px]" />
          <Skeleton className="h-[280px] rounded-[24px]" />
        </div>
        <Skeleton className="h-[260px] rounded-[24px]" />
      </div>
    );
  }

  const hasData = profile && (
    profile.stats?.total_sessions > 0 ||
    profile.stats?.total_answers > 0 ||
    (profile.weak_points || []).length > 0 ||
    (profile.strong_points || []).length > 0
  );

  if (!hasData) {
    return (
      <div className={PAGE_CLASS}>
        <div className="text-3xl font-display font-bold">个人画像</div>
        <Card className="mt-5 overflow-hidden border-primary/20 bg-[linear-gradient(135deg,rgba(245,158,11,0.12),rgba(20,184,166,0.08))] dark:bg-[linear-gradient(135deg,rgba(245,158,11,0.16),rgba(8,145,178,0.12))]">
          <CardContent className="p-8 md:p-10">
            <div className="max-w-2xl">
              <Badge className="mb-4 bg-primary/12 text-primary">还没有训练数据</Badge>
              <div className="text-2xl font-semibold leading-tight md:text-4xl">
                先积累几轮回答，再让页面开始提炼真正的重点。
              </div>
              <div className="mt-4 text-sm leading-7 text-dim md:text-base">
                开始面试后，系统会逐步把你的弱项、强项、答题模式和领域变化沉淀下来。等第一批数据形成，页面会自动切到驾驶舱视图。
              </div>
              <Button variant="gradient" size="lg" className="mt-6" onClick={() => navigate("/")}>
                开始第一场面试
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const stats = profile.stats || {};
  const scoreHistory = stats.score_history || [];
  // 知识轴: weak_points / strong_points 现在只承载知识类。
  // 老数据里可能存在 axis=performance 的遗留条目,用 isKnowledgeAxis 过滤掉。
  const weakActive = (profile.weak_points || []).filter(
    (item) => !item.improved && !item.archived && isKnowledgeAxis(item)
  );
  const weakImproved = sortByDateDesc(
    (profile.weak_points || []).filter((item) => item.improved && isKnowledgeAxis(item)),
    "improved_at",
    "last_seen"
  );
  const knowledgeStrong = sortByDateDesc(
    (profile.strong_points || []).filter(isKnowledgeAxis),
    "first_seen",
    "first_seen"
  );
  const thinkingStrengths = profile.thinking_patterns?.strengths || [];
  const thinkingGaps = profile.thinking_patterns?.gaps || [];
  const communicationHabits = profile.communication?.habits || [];
  const communicationSuggestions = profile.communication?.suggestions || [];
  const masteryMap = profile.topic_mastery || {};
  const realTopicSet = getRealTopicSet(profile, scoreHistory, canonicalTopics);

  const priorityWeaknesses = buildPriorityWeaknesses(weakActive, masteryMap);

  // 表现轴: 全部从 behavior_signals 派生,不再从 weak_points 派生
  const behaviorView = buildBehaviorSignals(profile);
  const featuredBehavior = behaviorView.featured;
  const activePerfDims = behaviorView.namespaces.filter(
    (dim) =>
      (dim.negative?.length || 0) > 0 ||
      (dim.positive?.length || 0) > 0 ||
      (dim.improved?.length || 0) > 0
  );

  const domains = buildDomainInsights(profile, realTopicSet);
  const focusDomains = domains.filter((item) => item.zone === "focus");
  const buildDomains = domains.filter((item) => item.zone === "build");
  const strongDomains = domains.filter((item) => item.zone === "strong");
  const topicPriorities = [...focusDomains, ...buildDomains, ...strongDomains].map((item) => ({
    ...item,
    topWeakness: priorityWeaknesses.find((weakness) => weakness.topic === item.topic)?.point || "",
  }));
  const featuredTopic = topicPriorities[0] || null;
  const secondaryTopic = topicPriorities[1] || null;
  const extraTopicCount = Math.max(topicPriorities.length - 2, 0);
  const modeCounts = buildModeCounts(stats, scoreHistory);
  const trainingModeStats = buildTrainingModeStats(stats, scoreHistory);
  const latestEntry = getLatestEntry(scoreHistory);
  const trendDelta = getTrendDelta(scoreHistory);

  return (
    <div className={PAGE_CLASS}>
      <div className="animate-fade-in">
        <div className="text-3xl font-display font-bold tracking-tight md:text-4xl">个人画像</div>
        <div className="mt-2 text-sm text-dim">
          {stats.total_answers || 0} 次回答分析
          {stats.total_sessions ? ` | ${stats.total_sessions} 次完整面试` : ""}
          {profile.updated_at ? ` | 上次更新 ${formatMinute(profile.updated_at)}` : ""}
        </div>
      </div>

      <Card className="mt-5 animate-fade-in-up [animation-delay:0.04s]">
        <CardContent className="p-4 md:p-5">
          <SectionHeader icon={<TrendingUp size={18} />} title="练习统计" />

          <div className="mt-5 grid gap-6 lg:grid-cols-[auto_1px_1fr] items-center rounded-3xl border border-border/60 bg-black/[0.02] dark:bg-white/[0.02] p-5 md:p-6 lg:p-7 shadow-sm">
            <div className="flex gap-8 md:gap-14 lg:pl-2">
              <div className="flex flex-col gap-1.5">
                <div className="text-sm font-medium text-dim">总练习次数</div>
                <div className="mt-1 flex items-baseline gap-1.5">
                  <div className="text-4xl font-bold tracking-tight text-primary drop-shadow-sm">{stats.total_sessions || 0}</div>
                </div>
              </div>
              <div className="flex flex-col gap-1.5">
                <div className="text-sm font-medium text-dim">综合平均分</div>
                <div className="mt-1 flex items-baseline gap-1.5">
                  <div className="text-4xl font-bold tracking-tight text-green drop-shadow-sm">{stats.avg_score ?? "-"}</div>
                </div>
              </div>
            </div>

            <div className="h-full w-px bg-border/60 hidden lg:block" />

            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 md:gap-4 w-full lg:pl-6">
              {trainingModeStats.map((item) => (
                <div
                  key={item.mode}
                  className={cn(
                    "flex flex-col rounded-2xl border border-border/80 border-l-[4px] px-4 py-3 bg-[linear-gradient(180deg,rgba(255,255,255,0.72),rgba(255,255,255,0.92))] dark:bg-[linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.04))]",
                    item.borderClassName,
                    item.glowClassName
                  )}
                >
                  <div className={cn("text-xs font-medium md:text-sm", item.accentClassName)}>{item.title}</div>
                  <div className="mt-2.5 flex items-baseline gap-3">
                    <div>
                      <span className={cn("text-xl font-semibold tracking-tight", item.accentClassName)}>{item.count}</span>
                      <span className="ml-0.5 text-[10px] text-dim">次</span>
                    </div>
                    <div className="text-border/60 text-xs">/</div>
                    <div>
                      <span className={cn("text-xl font-semibold tracking-tight", item.accentClassName)}>{item.avgScore ?? "-"}</span>
                      <span className="ml-0.5 text-[10px] text-dim">分</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ═══ 能力特征 (大卡, 知识轴) ═══ */}
      <Card className="mt-5 animate-fade-in-up [animation-delay:0.08s]">
        <CardContent className="p-5 md:p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-baseline gap-3">
              <div className="text-xl font-display font-bold tracking-tight">能力特征</div>
              <div className="text-xs text-dim">"你懂什么、会什么" — 技术知识维度</div>
            </div>
            <Badge variant="outline" className="rounded-full px-2.5 py-0.5 text-[10px]">知识轴</Badge>
          </div>

          {/* 焦点领域 */}
          <div className="mt-6">
            <SectionHeader
              icon={<Target size={18} />}
              title="焦点领域"
              caption="按训练领域排列，聚焦当前最该补的方向。"
              action={(
                <Button variant="outline" size="sm" onClick={() => navigate("/history")}>
                  查看全部记录
                </Button>
              )}
            />
            <div className="mt-5 space-y-4">
              {featuredTopic ? (
                <TopicPriorityCard
                  item={featuredTopic}
                  onSelect={(topic) => navigate(`/profile/topic/${topic}`)}
                  label="主推荐"
                />
              ) : (
                <div className="rounded-[24px] border border-dashed border-border/80 px-5 py-8 text-sm text-dim">
                  目前没有可继续追踪的真实训练领域。
                </div>
              )}
              {secondaryTopic && (
                <TopicPriorityCard
                  item={secondaryTopic}
                  onSelect={(topic) => navigate(`/profile/topic/${topic}`)}
                  label="次推荐"
                />
              )}
              {extraTopicCount > 0 && (
                <div className="rounded-2xl border border-border/70 bg-black/[0.02] px-4 py-3 text-xs leading-5 text-dim dark:bg-white/[0.02]">
                  还有 {extraTopicCount} 个领域在排队，完整列表见下方能力地图。
                </div>
              )}
            </div>
          </div>

          <div className="my-5 border-t border-border/60" />

          {/* 知识证据 */}
          <div>
            <SectionHeader
              icon={<Clock3 size={18} />}
              title="知识证据"
              caption="按弱点 / 强项 / 已改善分组的原始观察，可点击核对判断依据。"
            />
            <div className="mt-4">
              <EvidenceTable
                weakItems={priorityWeaknesses}
                strongItems={knowledgeStrong}
                improvedItems={weakImproved}
              />
            </div>
          </div>

          <div className="my-5 border-t border-border/60" />

          {/* 能力地图 */}
          <div>
            <SectionHeader
              icon={<Target size={18} />}
              title="能力地图"
              caption="覆盖到的真实训练主题与各自掌握度。"
            />
            <div className="mt-4">
              <DomainTable
                items={topicPriorities}
                onSelect={(topic) => navigate(`/profile/topic/${topic}`)}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ═══ 表现特征 (大卡, 表现轴) ═══ */}
      <Card className="mt-5 animate-fade-in-up [animation-delay:0.12s]">
        <CardContent className="p-5 md:p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-baseline gap-3">
              <div className="text-xl font-display font-bold tracking-tight">表现特征</div>
              <div className="text-xs text-dim">"你怎么表达、怎么推导" — 行为模式维度</div>
            </div>
            <Badge variant="outline" className="rounded-full px-2.5 py-0.5 text-[10px]">表现轴</Badge>
          </div>

          {/* 主推行为模式 + 四 namespace 摘要 */}
          <div className="mt-6">
            {featuredBehavior ? (
              <div className="rounded-[20px] border border-amber-500/20 bg-[linear-gradient(135deg,rgba(245,158,11,0.06),rgba(251,191,36,0.03))] p-5 md:p-6 dark:bg-[linear-gradient(135deg,rgba(245,158,11,0.10),rgba(251,191,36,0.04))]">
                <div className="inline-flex rounded-full bg-amber-500/12 px-3 py-1 text-xs font-medium text-amber-600 dark:text-amber-400">
                  最突出的行为模式
                </div>
                <div className="mt-3 text-lg font-semibold leading-relaxed md:text-xl">
                  {featuredBehavior.description || featuredBehavior.id}
                </div>
                <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-dim">
                  <span>{PERFORMANCE_DIMENSIONS[featuredBehavior.namespace]?.label || featuredBehavior.namespace}</span>
                  <span>·</span>
                  <span className="font-mono">{featuredBehavior.id}</span>
                  <span>·</span>
                  <span>出现 {featuredBehavior.times_seen || 1} 次</span>
                </div>
                {featuredBehavior.examples?.length > 0 && (
                  <div className="mt-3 rounded-xl border border-border/60 bg-card/90 px-3 py-2 text-xs leading-5 text-dim">
                    最近一次: {featuredBehavior.examples[featuredBehavior.examples.length - 1].snippet}
                  </div>
                )}
              </div>
            ) : (
              <div className="rounded-[20px] border border-dashed border-border/70 px-5 py-7 text-sm leading-6 text-dim">
                还没有累积到稳定的行为模式。完成下一次面试后，系统会按四个维度（推导 / 叙事 / 表达 / 元认知）开始识别你的模式。
              </div>
            )}

            {activePerfDims.length > 0 && (
              <div className="mt-5 grid grid-cols-2 gap-3 md:grid-cols-4">
                {behaviorView.namespaces.map((dim) => (
                  <PerformanceDimCard key={dim.key} dim={dim} />
                ))}
              </div>
            )}
          </div>

          {/* 完整 behavior_signals 列表 */}
          {behaviorView.activeNegativeCount + behaviorView.activePositiveCount + behaviorView.improvedCount > 0 && (
            <>
              <div className="my-5 border-t border-border/60" />
              <div>
                <SectionHeader
                  icon={<Brain size={18} />}
                  title="模式清单"
                  caption="按维度分组的所有 behavior_signals，点开行可看证据片段。"
                />
                <div className="mt-4">
                  <BehaviorSignalList namespaces={behaviorView.namespaces} />
                </div>
              </div>
            </>
          )}

          {/* 旧观察（合并 communication.* 与 thinking_patterns.*） */}
          {(profile.communication?.style ||
            communicationHabits.length > 0 ||
            communicationSuggestions.length > 0 ||
            thinkingGaps.length > 0 ||
            thinkingStrengths.length > 0) && (
            <>
              <div className="my-5 border-t border-border/60" />
              <CollapsibleSection
                title="旧观察聚合"
                caption="来自历史 session 的自由文本聚合。新数据流已切到上方 behavior_signals，这里保留作为参考。"
                defaultOpen={behaviorView.activeNegativeCount === 0}
                badge={<Badge variant="outline" className="text-[10px]">legacy</Badge>}
              >
                <div className="space-y-5">
                  {profile.communication?.style && (
                    <div className="rounded-2xl bg-black/[0.02] p-4 dark:bg-white/[0.02]">
                      <div className="text-xs font-medium text-dim mb-1">沟通风格</div>
                      <div className="text-sm leading-7">{profile.communication.style}</div>
                    </div>
                  )}
                  {communicationHabits.length > 0 && (
                    <div>
                      <div className="text-xs font-medium text-dim mb-2">表达习惯</div>
                      <HabitTagList items={communicationHabits} />
                    </div>
                  )}
                  {(thinkingGaps.length > 0 || thinkingStrengths.length > 0 || communicationSuggestions.length > 0) && (
                    <div className="grid gap-x-6 gap-y-4 sm:grid-cols-2 lg:grid-cols-3">
                      <PatternColumn title="风险" color="text-red" items={thinkingGaps} />
                      <PatternColumn title="优势" color="text-green" items={thinkingStrengths} />
                      <PatternColumn title="训练建议" color="text-primary" items={communicationSuggestions} />
                    </div>
                  )}
                </div>
              </CollapsibleSection>
            </>
          )}
        </CardContent>
      </Card>

      {/* ═══ 辅助底栏: 训练结构 + 最近评分 + 趋势 ═══ */}
      <Card className="mt-5 animate-fade-in-up [animation-delay:0.16s]">
        <CardContent className="p-5 md:p-6">
          <SectionHeader
            icon={<Activity size={18} />}
            title="训练结构"
            caption="模式分布、最近评分、趋势。"
          />
          <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="space-y-3">
              {modeCounts.length > 0 ? modeCounts.map((item) => (
                <div key={item.mode}>
                  <div className="flex items-center justify-between gap-3 text-sm">
                    <span>{item.label}</span>
                    <span className="text-dim">{item.count} 次</span>
                  </div>
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-border">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${item.percent}%`, backgroundColor: item.color }}
                    />
                  </div>
                </div>
              )) : (
                <div className="rounded-xl border border-dashed border-border/80 px-3 py-4 text-sm text-dim">
                  暂无训练分布数据。
                </div>
              )}
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-2xl border border-border/80 bg-card/80 p-4">
                <div className="text-xs font-medium text-dim">最近一次评分</div>
                <div className="mt-2 text-2xl font-semibold">
                  {latestEntry?.avg_score != null ? `${latestEntry.avg_score}/10` : "--"}
                </div>
                <div className="mt-2 text-xs text-dim">
                  {latestEntry ? `${(MODE_META[latestEntry.mode] || MODE_META.topic_drill).label} · ${formatShortDate(latestEntry.date)}` : "暂无评分记录"}
                </div>
              </div>
              <div className="rounded-2xl border border-border/80 bg-card/80 p-4">
                <div className="text-xs font-medium text-dim">趋势变化</div>
                <div className={cn(
                  "mt-2 text-2xl font-semibold",
                  trendDelta == null ? "text-text" : trendDelta >= 0 ? "text-green" : "text-red"
                )}>
                  {trendDelta == null ? "--" : trendDelta > 0 ? `+${trendDelta}` : trendDelta}
                </div>
                <div className="mt-2 text-xs text-dim">相比上一条评分记录</div>
              </div>
              <div className="rounded-2xl bg-black/4 px-4 py-3 dark:bg-white/[0.04]">
                <div className="text-xs text-dim">回答分析</div>
                <div className="mt-1 text-xl font-semibold">{stats.total_answers || 0}</div>
              </div>
              <div className="rounded-2xl bg-black/4 px-4 py-3 dark:bg-white/[0.04]">
                <div className="text-xs text-dim">覆盖主题</div>
                <div className="mt-1 text-xl font-semibold">{domains.length}</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {scoreHistory.length >= 2 && (
        <Card className="mt-5 animate-fade-in-up [animation-delay:0.2s]">
          <CardContent className="p-5 md:p-6">
            <SectionHeader
              icon={<TrendingUp size={18} />}
              title="成长趋势"
            />
            <div className="mt-5 rounded-[24px] border border-border/70 bg-black/[0.02] p-3 dark:bg-white/[0.02] md:p-4">
              <ScoreChart history={scoreHistory} />
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
