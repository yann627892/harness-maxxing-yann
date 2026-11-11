import * as React from 'react'
import { Globe, KeyRound, PlugZap, PlusCircle, RefreshCw, Terminal, Trash2 } from 'lucide-react'
import { useClaudeAuth, useClaudeAuthRemove, useClaudeSettings, useClaudeStatus, useClaudeTest } from '@/lib/queries'
import { errorMessage } from '@/lib/api'
import { fmtRelative } from '@/lib/format'
import type { ClaudeAuthMethod, ClaudeGatewayProvider, ClaudeProvider, ClaudeTestResult } from '@/lib/types'
import { useLang, useT, type TKey } from '@/i18n'
import { Badge, Button, CodeBlock, FormField, Input, SecretInput, Select, Sensitive, Skeleton, Tabs, TabsContent, TabsList, TabsTrigger, confirm } from '@/components/ui'
import { Callout, Steps } from '../common'

const METHOD: Record<NonNullable<ClaudeAuthMethod>, TKey> = {
  oauth_token: 'claude.method.oauth_token',
  credentials: 'claude.method.credentials',
  api_key: 'claude.method.api_key',
  gateway: 'claude.method.gateway',
}

/** Gateway presets, in the same order the server knows them (maxxing/gateway.py). */
const GW: { id: ClaudeGatewayProvider; label: TKey; desc: TKey; placeholder: string; keyUrl?: string }[] = [
  { id: 'zen', label: 'claude.gw.zen', desc: 'claude.gw.zenDesc', placeholder: 'oc_sk…', keyUrl: 'https://opencode.ai/console' },
  { id: 'zen_go', label: 'claude.gw.zen_go', desc: 'claude.gw.zen_goDesc', placeholder: 'oc_sk…', keyUrl: 'https://opencode.ai/console' },
  { id: 'openrouter', label: 'claude.gw.openrouter', desc: 'claude.gw.openrouterDesc', placeholder: 'sk-or…', keyUrl: 'https://openrouter.ai/settings/keys' },
  { id: 'custom', label: 'claude.gw.custom', desc: 'claude.gw.customDesc', placeholder: '…' },
]

/** Live status line + test button. */
export function ClaudeStatusCard({ onReplace, onAdd }: { onReplace?: () => void; onAdd?: 'oauth_token' | 'gateway' }) {
  const t = useT()
  const lang = useLang()
  const { data: s, isLoading } = useClaudeStatus()
  const test = useClaudeTest()
  const remove = useClaudeAuthRemove()
  const settings = useClaudeSettings()
  const [result, setResult] = React.useState<ClaudeTestResult | null>(null)
  if (isLoading || !s) return <Skeleton className="h-[74px] rounded-lg" />
  if (!s.installed)
    return (
      <Callout tone="error" title={t('claude.notInstalled')}>
        {t('claude.notInstalledDesc')} <CodeBlock inline code="maxxing doctor" className="mt-2" />
      </Callout>
    )
  const dual = !!s.gateway && !!s.anthropic?.connected      // both sides: each session picks with its model
  return (
    <div className="grid gap-3">
      <Callout
        tone={s.auth.ok ? 'success' : 'warning'}
        title={s.auth.ok ? t('claude.connected') : s.auth.method ? t('claude.signInExpired') : t('claude.notConnected')}
        action={
          s.auth.ok ? (
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                loading={test.isPending}
                onClick={() =>
                  test.mutate(undefined, {
                    onSuccess: setResult,
                    onError: (e) => setResult({ ok: false, model: '', ms: 0, error: errorMessage(e) }),
                  })
                }
              >
                <PlugZap /> {t('claude.test')}
              </Button>
            </div>
          ) : undefined
        }
      >
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
          {s.auth.method && <Badge variant="outline">{t(METHOD[s.auth.method])}</Badge>}
          {s.version && <span>Claude Code {s.version}</span>}
          {s.auth.detail && (
            <span>
              · <Sensitive>{s.auth.detail}</Sensitive>
            </span>
          )}
          {s.gateway && !dual && <span className="text-muted-foreground">· {s.gateway.base_url}</span>}
          {s.auth.ok && s.auth.checked_at && (
            <span className="text-muted-foreground">· {t('claude.verified', { when: fmtRelative(s.auth.checked_at, lang) })}</span>
          )}
        </span>
      </Callout>
      {dual && (
        <div className="grid gap-3 rounded-lg border p-4">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
            <Badge variant="outline">{t('cconn.anthropicSide')}</Badge>
            {s.anthropic?.detail && <Sensitive>{s.anthropic.detail}</Sensitive>}
            <span className="text-muted-foreground">·</span>
            <Badge variant="outline">{s.gateway!.name}</Badge>
            <span className="text-muted-foreground">{s.gateway!.base_url}</span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium">{t('cconn.defaultProvider')}</span>
            <Select
              value={s.default_provider || 'gateway'}
              onValueChange={(v) => settings.mutate({ claude_provider: v as ClaudeProvider })}
              aria-label={t('cconn.defaultProvider')}
              options={[
                { value: 'anthropic', label: t('cconn.anthropicSide') },
                { value: 'gateway', label: s.gateway!.name },
              ]}
            />
          </div>
          <p className="text-sm text-muted-foreground">{t('cconn.defaultProviderDesc')}</p>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              loading={test.isPending}
              onClick={() =>
                test.mutate('anthropic', {
                  onSuccess: setResult,
                  onError: (e) => setResult({ ok: false, model: '', ms: 0, error: errorMessage(e) }),
                })
              }
            >
              <PlugZap /> {t('cconn.testAnthropic')}
            </Button>
            <Button
              variant="outline"
              size="sm"
              loading={test.isPending}
              onClick={() =>
                test.mutate('gateway', {
                  onSuccess: setResult,
                  onError: (e) => setResult({ ok: false, model: '', ms: 0, error: errorMessage(e) }),
                })
              }
            >
              <PlugZap /> {t('cconn.testGateway')}
            </Button>
          </div>
        </div>
      )}
      {result && (
        <Callout tone={result.ok ? 'success' : 'error'} title={result.ok ? t('claude.testOk') : t('claude.testFailed')}>
          {result.ok ? t('claude.testOkDesc', { model: result.model, ms: Math.round(result.ms) }) : result.error}
        </Callout>
      )}
      {s.auth.ok && (onReplace || onAdd) && (
        <div className="flex flex-wrap gap-2">
          {onReplace && (
            <Button variant="outline" size="sm" onClick={onReplace}>
              <RefreshCw /> {t('claude.replace')}
            </Button>
          )}
          {onAdd && (
            <Button variant="outline" size="sm" onClick={() => window.dispatchEvent(new CustomEvent('maxxing:add-credential', { detail: onAdd }))}>
              <PlusCircle /> {t(onAdd === 'gateway' ? 'cconn.addGateway' : 'cconn.addAnthropic')}
            </Button>
          )}
          {dual ? (
            <>
              <Button
                variant="ghost"
                size="sm"
                className="text-destructive hover:text-destructive"
                loading={remove.isPending}
                onClick={async () => {
                  if (await confirm({ title: t('cconn.removeGwTitle'), description: t('cconn.removeGwDesc'), destructive: true, confirmText: t('common.remove') })) remove.mutate('gateway')
                }}
              >
                <Trash2 /> {t('cconn.removeGateway')}
              </Button>
              {s.anthropic?.method !== 'credentials' && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-destructive hover:text-destructive"
                  loading={remove.isPending}
                  onClick={async () => {
                    if (await confirm({ title: t('cconn.removeAnTitle'), description: t('cconn.removeAnDesc'), destructive: true, confirmText: t('common.remove') })) remove.mutate('anthropic')
                  }}
                >
                  <Trash2 /> {t('cconn.removeAnthropic')}
                </Button>
              )}
            </>
          ) : (
            s.auth.method !== 'credentials' && (
              <Button
                variant="ghost"
                size="sm"
                className="text-destructive hover:text-destructive"
                loading={remove.isPending}
                onClick={async () => {
                  if (await confirm({ title: t('claude.removeTitle'), description: t('claude.removeDesc'), destructive: true, confirmText: t('common.remove') })) remove.mutate()
                }}
              >
                <Trash2 /> {t('claude.remove')}
              </Button>
            )
          )}
        </div>
      )}
    </div>
  )
}

/** Paste a `claude setup-token` OAuth token, an Anthropic API key, or a gateway key (OpenCode Zen /
 *  Go, OpenRouter, custom Anthropic-compatible endpoint), then the server tests it. */
export function ClaudeAuthForm({ onDone, initialKind }: { onDone?: () => void; initialKind?: 'oauth_token' | 'api_key' | 'gateway' }) {
  const t = useT()
  const [kind, setKind] = React.useState<'oauth_token' | 'api_key' | 'gateway'>(initialKind || 'oauth_token')
  const [provider, setProvider] = React.useState<ClaudeGatewayProvider>('zen')
  const [baseUrl, setBaseUrl] = React.useState('')
  const [value, setValue] = React.useState('')
  const [error, setError] = React.useState<string | null>(null)
  const auth = useClaudeAuth()
  const gw = GW.find((g) => g.id === provider) || GW[0]
  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    const v = value.trim()
    const url = baseUrl.trim()
    if (!v) return setError(kind === 'gateway' ? t('claude.gwKeyFirst') : t('claude.pasteFirst'))
    if (kind === 'gateway' && provider === 'custom' && !url) return setError(t('claude.gatewayUrlFirst'))
    setError(null)
    auth.mutate(
      kind === 'gateway' ? { kind, value: v, provider, ...(provider === 'custom' ? { base_url: url } : {}) } : { kind, value: v },
      {
        onSuccess: (r) => {
          if (r.ok) {
            setValue('')
            onDone?.()
          } else setError(r.error || t('claude.authFailed'))
        },
        onError: (e) => setError(errorMessage(e)),
      },
    )
  }
  return (
    <Tabs value={kind} onValueChange={(v) => {
        setKind(v as typeof kind)
        setError(null)
      }}>
      <TabsList className="w-full sm:w-auto">
        <TabsTrigger value="oauth_token" className="flex-1">
          <Terminal /> {t('claude.tabSubscription')}
        </TabsTrigger>
        <TabsTrigger value="api_key" className="flex-1">
          <KeyRound /> {t('claude.tabApiKey')}
        </TabsTrigger>
        <TabsTrigger value="gateway" className="flex-1">
          <Globe /> {t('claude.tabGateway')}
        </TabsTrigger>
      </TabsList>
      <form onSubmit={submit} className="mt-4 grid gap-5" noValidate>
        <TabsContent value="oauth_token" className="mt-0 grid gap-4">
          <p className="text-sm text-muted-foreground">{t('claude.subscriptionIntro')}</p>
          <Steps
            items={[
              <div className="grid gap-2">
                <span>{t('claude.step1')}</span>
                <CodeBlock inline code={`ssh root@${window.location.hostname}`} />
              </div>,
              <div className="grid gap-2">
                <span>{t('claude.step2')}</span>
                <CodeBlock inline code="claude setup-token" />
              </div>,
              <span>{t('claude.step3')}</span>,
            ]}
          />
        </TabsContent>
        <TabsContent value="api_key" className="mt-0 grid gap-4">
          <p className="text-sm text-muted-foreground">{t('claude.apiKeyIntro')}</p>
          <Steps
            items={[
              <span>
                {t('claude.apiStep1')}{' '}
                <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener noreferrer">
                  console.anthropic.com
                </a>
              </span>,
              <span>{t('claude.apiStep2')}</span>,
            ]}
          />
        </TabsContent>
        <TabsContent value="gateway" className="mt-0 grid gap-4">
          <p className="text-sm text-muted-foreground">{t('claude.gatewayIntro')}</p>
          <FormField label={t('claude.gatewayProvider')}>
            <Select
              value={provider}
              onValueChange={(v) => setProvider(v as ClaudeGatewayProvider)}
              aria-label={t('claude.gatewayProvider')}
              options={GW.map((g) => ({ value: g.id, label: `${t(g.label)} · ${t(g.desc)}` }))}
            />
          </FormField>
          {provider === 'custom' ? (
            <FormField label={t('claude.gatewayUrlLabel')} hint={t('claude.gatewayUrlHint')}>
              <Input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://gw.example.com/anthropic" inputMode="url" autoComplete="off" />
            </FormField>
          ) : (
            <Steps
              items={[
                <span>
                  {t(provider === 'openrouter' ? 'claude.apiStep1' : 'claude.gwZenStep1')}{' '}
                  <a href={gw.keyUrl} target="_blank" rel="noopener noreferrer">
                    {gw.keyUrl?.replace('https://', '')}
                  </a>
                </span>,
                <span>{t(provider === 'openrouter' ? 'claude.gwOrStep2' : 'claude.gwZenStep2')}</span>,
              ]}
            />
          )}
        </TabsContent>
        <FormField
          label={kind === 'oauth_token' ? t('claude.tokenLabel') : kind === 'gateway' ? t('claude.gwKeyLabel') : t('claude.keyLabel')}
          error={error || undefined}
          hint={t('claude.storedHint')}
        >
          <SecretInput
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={kind === 'oauth_token' ? 'sk-ant-oat01-…' : kind === 'gateway' ? gw.placeholder : 'sk-ant-api03-…'}
          />
        </FormField>
        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" loading={auth.isPending}>
            <PlugZap /> {t('claude.connectTest')}
          </Button>
          {auth.isPending && <span className="text-[13px] text-muted-foreground">{t('claude.testing')}</span>}
        </div>
      </form>
    </Tabs>
  )
}

/** Status + (form when not connected or when adding / replacing). */
export function ClaudeConnect() {
  const t = useT()
  const { data: s } = useClaudeStatus()
  const [replacing, setReplacing] = React.useState(false)
  const [adding, setAdding] = React.useState<null | 'oauth_token' | 'gateway'>(null)
  React.useEffect(() => {
    const open = (ev: Event) => setAdding((ev as CustomEvent).detail)
    window.addEventListener('maxxing:add-credential', open)
    return () => window.removeEventListener('maxxing:add-credential', open)
  }, [])
  const gwOn = !!s?.gateway
  const anOn = !!s?.anthropic?.connected
  const add: null | 'oauth_token' | 'gateway' = !s?.installed || (gwOn && anOn) ? null : anOn ? 'gateway' : gwOn ? 'oauth_token' : null
  const showForm = !!s && s.installed && (!s.auth.ok || replacing || !!adding)
  return (
    <div className="grid gap-5">
      <ClaudeStatusCard onReplace={replacing || adding ? undefined : () => setReplacing(true)} onAdd={add || undefined} />
      {showForm && (
        <div className="rounded-lg border p-4 md:p-5">
          <ClaudeAuthForm key={adding || 'replace'} initialKind={adding || undefined} onDone={() => { setReplacing(false); setAdding(null) }} />
          {(replacing || adding) && (
            <Button variant="ghost" size="sm" className="mt-3" onClick={() => { setReplacing(false); setAdding(null) }}>
              {t('common.cancel')}
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
