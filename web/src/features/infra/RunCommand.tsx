import * as React from 'react'
import { Play, Terminal } from 'lucide-react'
import { useInfra, useInfraExec } from '@/lib/queries'
import { errorMessage } from '@/lib/api'
import { useT } from '@/i18n'
import { Button, Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, FormField, Select, Textarea, toast } from '@/components/ui'

export function RunCommand({ defaultTarget = 'local' }: { defaultTarget?: string }) {
  const t = useT()
  const v = useInfra().data
  const exec = useInfraExec()
  const [open, setOpen] = React.useState(false)
  const [target, setTarget] = React.useState(defaultTarget)
  const [command, setCommand] = React.useState('')
  const [result, setResult] = React.useState<{ stdout: string; stderr: string; exit_code: number; duration_ms: number } | null>(null)

  const servers = (v?.resources || []).filter((r) => r.kind === 'server').map((r) => r.details?.server as string).filter(Boolean)

  const run = (e: React.FormEvent) => {
    e.preventDefault()
    if (!command.trim()) return
    setResult(null)
    exec.mutate({ target, command: command.trim() }, {
      onSuccess: (r) => setResult(r),
      onError: (e) => toast.error(errorMessage(e)),
    })
  }

  const close = () => { setOpen(false); setCommand(''); setResult(null); exec.reset() }

  return (
    <>
      <Button variant="outline" size="sm" onClick={() => setOpen(true)} data-testid="infra-run-command">
        <Terminal /> <span className="max-[429px]:sr-only">{t('infra.run.title')}</span>
      </Button>
      <Dialog open={open} onOpenChange={(o) => !o && close()}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>{t('infra.run.title')}</DialogTitle>
            <DialogDescription>{t('infra.run.desc')}</DialogDescription>
          </DialogHeader>
          <form onSubmit={run} className="grid gap-3">
            <FormField label={t('infra.run.target')}>
              <Select
                value={target}
                onValueChange={setTarget}
                options={[
                  { value: 'local', label: t('infra.run.local') },
                  ...servers.map((s) => ({ value: `server:${s}`, label: s })),
                ]}
              />
            </FormField>
            <FormField label={t('infra.run.command')}>
              <Textarea value={command} onChange={(e) => setCommand(e.target.value)} rows={4} className="font-mono text-xs" placeholder="uptime && df -h" autoFocus />
            </FormField>
            <DialogFooter>
              <Button type="submit" loading={exec.isPending} disabled={!command.trim()}>
                <Play /> {t('infra.run.run')}
              </Button>
            </DialogFooter>
          </form>
          {result && (
            <div className="grid gap-2">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>exit {result.exit_code}</span>
                <span>· {result.duration_ms} ms</span>
              </div>
              {result.stdout && <pre className="max-h-64 overflow-auto rounded-md bg-muted/60 p-3 font-mono text-xs whitespace-pre-wrap">{result.stdout}</pre>}
              {result.stderr && <pre className="max-h-64 overflow-auto rounded-md bg-destructive/10 p-3 font-mono text-xs text-destructive whitespace-pre-wrap">{result.stderr}</pre>}
              {!result.stdout && !result.stderr && <p className="text-xs text-muted-foreground">{t('infra.run.noOutput')}</p>}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}