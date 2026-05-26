import { useState, useEffect, useCallback } from 'react'
import { api } from '../../api'
import type { AppSettings } from '../../types'

interface Props {
  onClose: () => void
}

export function SettingsModal({ onClose }: Props) {
  const [settings, setSettings] = useState<AppSettings | null>(null)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    api.getSettings().then(setSettings).catch(console.error)
  }, [])

  const handleEscape = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose()
  }, [onClose])

  useEffect(() => {
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [handleEscape])

  const handleSave = async () => {
    if (!settings) return
    setSaving(true)
    try {
      await api.putSettings(settings)
      setSaved(true)
      setTimeout(() => { setSaved(false); onClose() }, 800)
    } catch (e) {
      console.error('Settings save failed:', e)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0,0,0,0.55)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 380,
          background: 'var(--panel)',
          border: '1px solid var(--border2)',
          borderRadius: 10,
          overflow: 'hidden',
          boxShadow: '0 24px 48px rgba(0,0,0,0.6)',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', padding: '10px 16px',
          borderBottom: '1px solid var(--border)',
        }}>
          <span style={{ flex: 1, fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>
            Settings
          </span>
          <button
            onClick={onClose}
            style={{ background: 'transparent', border: 'none', color: 'var(--text2)', cursor: 'pointer', fontSize: 18, lineHeight: 1, padding: '0 2px' }}
          >
            &#xd7;
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: '16px 16px 8px' }}>
          {!settings ? (
            <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text2)', fontSize: 12 }}>Loading…</div>
          ) : (
            <>
              <SettingRow label="Display units">
                <RadioGroup
                  options={[
                    { value: 'mm', label: 'mm (millimetres)' },
                    { value: 'cm', label: 'cm (centimetres)' },
                  ]}
                  value={settings.units}
                  onChange={(v) => setSettings({ ...settings, units: v })}
                />
              </SettingRow>

              <SettingRow label="Default velocity colormap">
                <RadioGroup
                  options={[
                    { value: 'RdBu_r', label: 'RdBu_r  (blue → white → red)' },
                    { value: 'viridis', label: 'viridis  (purple → green → yellow)' },
                  ]}
                  value={settings.default_colormap_velocity}
                  onChange={(v) => setSettings({ ...settings, default_colormap_velocity: v })}
                />
              </SettingRow>

              <SettingRow label="Default basemap">
                <RadioGroup
                  options={[
                    { value: 'esri_satellite', label: 'ESRI Satellite' },
                    { value: 'osm', label: 'OpenStreetMap' },
                  ]}
                  value={settings.default_basemap}
                  onChange={(v) => setSettings({ ...settings, default_basemap: v })}
                />
              </SettingRow>
            </>
          )}
        </div>

        {/* Footer */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, padding: '8px 16px 14px' }}>
          <button
            onClick={onClose}
            style={{
              height: 30, padding: '0 16px', background: 'transparent',
              border: '1px solid var(--border2)', borderRadius: 6,
              color: 'var(--text2)', cursor: 'pointer', fontSize: 12,
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving || saved || !settings}
            style={{
              height: 30, padding: '0 16px',
              background: saved ? 'var(--accent2)' : 'rgba(41,182,246,0.15)',
              border: `1px solid ${saved ? 'var(--accent2)' : 'var(--accent)'}`,
              borderRadius: 6,
              color: saved ? '#000' : 'var(--accent)',
              cursor: saving || !settings ? 'not-allowed' : 'pointer',
              fontSize: 12, fontWeight: 600,
              transition: 'all 0.15s',
            }}
          >
            {saved ? 'Saved ✓' : saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}

function SettingRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.6px', color: 'var(--text2)', marginBottom: 8 }}>
        {label}
      </div>
      {children}
    </div>
  )
}

function RadioGroup({ options, value, onChange }: {
  options: { value: string; label: string }[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {options.map((opt) => (
        <label key={opt.value} style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
          <input
            type="radio"
            checked={value === opt.value}
            onChange={() => onChange(opt.value)}
            style={{ accentColor: 'var(--accent)', width: 13, height: 13, flexShrink: 0 }}
          />
          <span style={{ fontSize: 12, color: value === opt.value ? 'var(--text)' : 'var(--text2)' }}>
            {opt.label}
          </span>
        </label>
      ))}
    </div>
  )
}
