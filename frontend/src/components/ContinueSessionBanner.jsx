import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Clock, PlayCircle } from "lucide-react";
import { getInProgressSessions } from "../api/interview";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

function formatRelative(iso) {
  if (!iso) return "";
  const t = new Date(iso).getTime();
  const now = Date.now();
  const diff = Math.max(0, now - t);
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "刚刚";
  if (mins < 60) return `${mins} 分钟前`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  return `${days} 天前`;
}

export default function ContinueSessionBanner({ mode, title = "继续上次训练" }) {
  const navigate = useNavigate();
  const [session, setSession] = useState(null);

  useEffect(() => {
    let cancelled = false;
    getInProgressSessions(mode)
      .then((data) => {
        if (cancelled) return;
        setSession(data.items?.[0] || null);
      })
      .catch(() => {
        if (!cancelled) setSession(null);
      });
    return () => {
      cancelled = true;
    };
  }, [mode]);

  if (!session) return null;

  const subtitle =
    session.mode === "jd_prep"
      ? [session.meta?.company, session.meta?.position].filter(Boolean).join(" · ") || "JD 备面"
      : session.topic || "综合";
  const progressLabel =
    session.questions_count > 0
      ? `${session.answered_count}/${session.questions_count} 已答`
      : session.answered_count > 0
        ? `${session.answered_count} 轮对话`
        : "尚未作答";

  return (
    <Card className="mb-6 border-primary/30 bg-primary/5 backdrop-blur-sm">
      <CardContent className="p-4 md:p-5 flex flex-col md:flex-row md:items-center gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            <Badge variant="outline" className="border-primary/40 bg-primary/10 text-primary font-medium">
              {title}
            </Badge>
            <span className="text-[13px] text-dim font-medium">{progressLabel}</span>
          </div>
          <div className="text-[15px] font-semibold text-text truncate">{subtitle}</div>
          <div className="flex items-center gap-1.5 text-[12px] text-dim mt-1">
            <Clock size={12} />
            <span>更新于 {formatRelative(session.updated_at)}</span>
          </div>
        </div>
        <Button
          variant="gradient"
          className="shrink-0 rounded-xl px-6"
          onClick={() => navigate(`/interview/${session.session_id}`)}
        >
          <PlayCircle size={16} className="mr-1.5" /> 继续
        </Button>
      </CardContent>
    </Card>
  );
}
