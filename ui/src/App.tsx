import { useState, useEffect } from 'react'
import './App.css'

// Types
interface Player {
  id: string
  x: number
  y: number
  role: string
  possession: boolean
}

interface PlayData {
  time: number
  players: Player[]
  ball: { x: number; y: number }
}

interface Insight {
  top_decision: string
  confidence: string
  zones: number[][]
  details: {
    action: string
    zone: string
    location: { x: number; y: number }
    opportunity: number
    xt_improvement: number
  }
}

interface Counterfactual {
  player: string
  change: string
  lift: number
  valid: boolean
  violation?: string
  trajectory?: Array<{ x: number; y: number }>
}

interface CounterfactualPayload {
  counterfactuals: Counterfactual[]
  cvs: number
  n_valid: number
  n_total: number
}

interface LabelPrompt {
  play_id: string
  decision: string
  confidence: string
  flagged_reason: string
  ui_config: {
    show_modal: boolean
    buttons: Array<{ id: string; label: string; color: string }>
    tag_suggestions: string[]
  }
}

interface LiveFrame {
  timestamp: number
  frame_id: number
  xT_value: number
  fps: number
  overlay_url?: string
}

interface ClipInfo {
  clip_id: string
  timestamp: number
  xT_value: number
  duration: number
  file_path: string
}

function App() {
  const [playData, setPlayData] = useState<PlayData | null>(null)
  const [insight, setInsight] = useState<Insight | null>(null)
  const [labelPrompt, setLabelPrompt] = useState<LabelPrompt | null>(null)
  const [timeSlider, setTimeSlider] = useState(0)
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [customTag, setCustomTag] = useState('')

  // Counterfactual state
  const [counterfactuals, setCounterfactuals] = useState<CounterfactualPayload | null>(null)
  const [selectedCounterfactual, setSelectedCounterfactual] = useState<number | null>(null)
  const [delaySlider, setDelaySlider] = useState(0)
  const [showTrajectories, setShowTrajectories] = useState(true)

  // Live mode state
  const [liveMode, setLiveMode] = useState(false)
  const [liveFrame, setLiveFrame] = useState<LiveFrame | null>(null)
  const [videoSrc, setVideoSrc] = useState<string | null>(null)
  const [clips, setClips] = useState<ClipInfo[]>([])
  const [showOverlay, setShowOverlay] = useState(true)

  // Field dimensions
  const FIELD_WIDTH = 105
  const FIELD_HEIGHT = 68
  const SVG_WIDTH = 800
  const SVG_HEIGHT = (SVG_WIDTH * FIELD_HEIGHT) / FIELD_WIDTH

  // Scale coordinates
  const scaleX = (x: number) => (x / FIELD_WIDTH) * SVG_WIDTH
  const scaleY = (y: number) => SVG_HEIGHT - (y / FIELD_HEIGHT) * SVG_HEIGHT

  // Load sample data (in real app, this would come from backend)
  useEffect(() => {
    loadSamplePlay()
  }, [])

  const loadSamplePlay = () => {
    // Sample play data
    const samplePlay: PlayData = {
      time: 0.0,
      players: [
        { id: '1', x: 50, y: 25, role: 'QB', possession: true },
        { id: '2', x: 65, y: 15, role: 'WR', possession: false },
        { id: '3', x: 70, y: 35, role: 'WR', possession: false },
        { id: '4', x: 45, y: 30, role: 'RB', possession: false },
        { id: '5', x: 35, y: 28, role: 'LB', possession: false },
        { id: '6', x: 55, y: 45, role: 'CB', possession: false },
      ],
      ball: { x: 50, y: 25 }
    }

    // Sample insight
    const sampleInsight: Insight = {
      top_decision: 'Attack left: +31% xT',
      confidence: 'Medium',
      zones: Array(105).fill(0).map(() => Array(68).fill(0.5)),
      details: {
        action: 'Attack',
        zone: 'left',
        location: { x: 65, y: 15 },
        opportunity: 0.78,
        xt_improvement: 31
      }
    }

    // Sample counterfactuals
    const sampleCounterfactuals: CounterfactualPayload = {
      counterfactuals: [
        {
          player: 'WR',
          change: 'Delay 0.5s',
          lift: -0.12,
          valid: true,
          trajectory: [
            { x: 65, y: 15 },
            { x: 65, y: 15 },
            { x: 67, y: 16 },
            { x: 69, y: 17 },
            { x: 72, y: 18 }
          ]
        },
        {
          player: 'WR',
          change: 'Speed -10%',
          lift: -0.08,
          valid: true,
          trajectory: [
            { x: 65, y: 15 },
            { x: 66, y: 15 },
            { x: 68, y: 16 },
            { x: 70, y: 17 }
          ]
        },
        {
          player: 'WR',
          change: 'Angle +15°',
          lift: 0.15,
          valid: true,
          trajectory: [
            { x: 65, y: 15 },
            { x: 67, y: 13 },
            { x: 70, y: 11 },
            { x: 73, y: 10 }
          ]
        }
      ],
      cvs: 0.967,
      n_valid: 3,
      n_total: 3
    }

    setPlayData(samplePlay)
    setInsight(sampleInsight)
    setCounterfactuals(sampleCounterfactuals)
  }

  const handleLabelSubmit = (outcome: string) => {
    if (!labelPrompt) return

    const label = {
      play_id: labelPrompt.play_id,
      outcome,
      tags: selectedTags,
      timestamp: new Date().toISOString()
    }

    console.log('Label submitted:', label)
    alert(`Label saved: ${outcome}\nTags: ${selectedTags.join(', ')}`)
    setLabelPrompt(null)
    setSelectedTags([])
  }

  const toggleTag = (tag: string) => {
    if (selectedTags.includes(tag)) {
      setSelectedTags(selectedTags.filter(t => t !== tag))
    } else {
      setSelectedTags([...selectedTags, tag])
    }
  }

  const addCustomTag = () => {
    if (customTag && !selectedTags.includes(customTag)) {
      setSelectedTags([...selectedTags, customTag])
      setCustomTag('')
    }
  }

  const handleDelayChange = (value: number) => {
    setDelaySlider(value)
    // In real app, would trigger re-simulation with new delay
    console.log(`Delay changed to ${value}s`)
  }

  if (!playData || !insight) {
    return <div className="container">Loading...</div>
  }

  const selectedCf = selectedCounterfactual !== null && counterfactuals
    ? counterfactuals.counterfactuals[selectedCounterfactual]
    : null

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <h1>FieldSense AI v3.0</h1>
        <div className="confidence-badge" data-level={insight.confidence.toLowerCase()}>
          {insight.confidence} Confidence
        </div>
      </header>

      {/* Main Decision */}
      <div className="decision-panel">
        <h2 className="decision-text">{insight.top_decision}</h2>
        <div className="decision-details">
          <span className="detail-item">
            <strong>Action:</strong> {insight.details.action}
          </span>
          <span className="detail-item">
            <strong>Zone:</strong> {insight.details.zone}
          </span>
          <span className="detail-item">
            <strong>Opportunity:</strong> {(insight.details.opportunity * 100).toFixed(0)}%
          </span>
          {counterfactuals && (
            <span className="detail-item cvs-badge">
              <strong>CVS:</strong> {(counterfactuals.cvs * 100).toFixed(1)}%
            </span>
          )}
          {liveFrame && (
            <span className="detail-item fps-indicator">
              <strong>FPS:</strong> {liveFrame.fps.toFixed(1)}
            </span>
          )}
        </div>
      </div>

      {/* Live Mode Toggle */}
      <div className="live-controls">
        <button
          className={`live-toggle ${liveMode ? 'active' : ''}`}
          onClick={() => setLiveMode(!liveMode)}
        >
          {liveMode ? '⏸ Pause Live' : '▶ Start Live'}
        </button>
        {liveMode && (
          <label className="overlay-toggle">
            <input
              type="checkbox"
              checked={showOverlay}
              onChange={(e) => setShowOverlay(e.target.checked)}
            />
            Show overlay
          </label>
        )}
        {liveFrame && (
          <div className="live-stats">
            Frame: {liveFrame.frame_id} | xT: {liveFrame.xT_value.toFixed(3)}
          </div>
        )}
      </div>

      {/* Live Video Player */}
      {liveMode && (
        <div className="video-container">
          <div className="video-wrapper">
            {videoSrc ? (
              <video
                src={videoSrc}
                autoPlay
                muted
                className="live-video"
              />
            ) : (
              <div className="video-placeholder">
                <p>Connecting to live stream...</p>
              </div>
            )}
            {showOverlay && (
              <canvas
                className="video-overlay"
                width="1920"
                height="1080"
              />
            )}
          </div>
          {clips.length > 0 && (
            <div className="clips-panel">
              <h4>Auto-Generated Clips</h4>
              <div className="clips-list">
                {clips.map((clip) => (
                  <div key={clip.clip_id} className="clip-item">
                    <div className="clip-header">
                      <span className="clip-id">{clip.clip_id}</span>
                      <span className="clip-xt">xT: {clip.xT_value.toFixed(3)}</span>
                    </div>
                    <div className="clip-details">
                      Duration: {clip.duration}s | Time: {clip.timestamp.toFixed(1)}s
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Field Canvas */}
      <div className="field-container">
        <svg
          width={SVG_WIDTH}
          height={SVG_HEIGHT}
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          className="field-svg"
        >
          {/* Field background */}
          <rect
            width={SVG_WIDTH}
            height={SVG_HEIGHT}
            fill="#2d5016"
            stroke="#fff"
            strokeWidth="2"
          />

          {/* Field markings */}
          <line
            x1={SVG_WIDTH / 2}
            y1="0"
            x2={SVG_WIDTH / 2}
            y2={SVG_HEIGHT}
            stroke="#fff"
            strokeWidth="1"
            opacity="0.5"
          />

          {/* Opportunity zone highlight */}
          {insight.details.location && (
            <circle
              cx={scaleX(insight.details.location.x)}
              cy={scaleY(insight.details.location.y)}
              r="40"
              fill="rgba(255, 215, 0, 0.3)"
              stroke="gold"
              strokeWidth="2"
            />
          )}

          {/* Counterfactual trajectory */}
          {showTrajectories && selectedCf && selectedCf.trajectory && (
            <g className="trajectory">
              {/* Draw path */}
              <polyline
                points={selectedCf.trajectory
                  .map(p => `${scaleX(p.x)},${scaleY(p.y)}`)
                  .join(' ')}
                fill="none"
                stroke={selectedCf.lift > 0 ? '#4CAF50' : '#f44336'}
                strokeWidth="3"
                strokeDasharray="5,5"
                opacity="0.8"
              />
              {/* Draw trajectory points */}
              {selectedCf.trajectory.map((pos, idx) => (
                <circle
                  key={idx}
                  cx={scaleX(pos.x)}
                  cy={scaleY(pos.y)}
                  r="3"
                  fill={selectedCf.lift > 0 ? '#4CAF50' : '#f44336'}
                  opacity={0.6 + (idx / selectedCf.trajectory!.length) * 0.4}
                />
              ))}
            </g>
          )}

          {/* Players */}
          {playData.players.map((player) => (
            <g key={player.id}>
              <circle
                cx={scaleX(player.x)}
                cy={scaleY(player.y)}
                r="8"
                fill={player.possession ? '#4CAF50' : '#f44336'}
                stroke="#fff"
                strokeWidth="2"
              />
              <text
                x={scaleX(player.x)}
                y={scaleY(player.y) + 4}
                textAnchor="middle"
                fill="#fff"
                fontSize="10"
                fontWeight="bold"
              >
                {player.role}
              </text>
            </g>
          ))}

          {/* Ball */}
          <circle
            cx={scaleX(playData.ball.x)}
            cy={scaleY(playData.ball.y)}
            r="4"
            fill="#FFC107"
            stroke="#fff"
            strokeWidth="1"
          />
        </svg>
      </div>

      {/* Counterfactual Controls */}
      {counterfactuals && (
        <div className="counterfactual-panel">
          <h4>What-If Scenarios</h4>

          <div className="counterfactual-list">
            {counterfactuals.counterfactuals.map((cf, idx) => (
              <div
                key={idx}
                className={`counterfactual-item ${selectedCounterfactual === idx ? 'selected' : ''} ${cf.lift > 0 ? 'positive' : 'negative'}`}
                onClick={() => setSelectedCounterfactual(idx)}
              >
                <div className="cf-header">
                  <span className="cf-player">{cf.player}</span>
                  <span className={`cf-lift ${cf.lift > 0 ? 'positive' : 'negative'}`}>
                    {cf.lift > 0 ? '+' : ''}{(cf.lift * 100).toFixed(1)}% xT
                  </span>
                </div>
                <div className="cf-change">{cf.change}</div>
                {!cf.valid && (
                  <div className="cf-invalid">⚠ {cf.violation}</div>
                )}
              </div>
            ))}
          </div>

          {/* Delay Slider */}
          <div className="delay-control">
            <label>
              Simulation Delay: {delaySlider.toFixed(2)}s
              <span className="cvs-indicator">
                {counterfactuals.n_valid}/{counterfactuals.n_total} valid
              </span>
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={delaySlider}
              onChange={(e) => handleDelayChange(parseFloat(e.target.value))}
              className="delay-slider"
            />
          </div>

          {/* Trajectory toggle */}
          <div className="trajectory-toggle">
            <label>
              <input
                type="checkbox"
                checked={showTrajectories}
                onChange={(e) => setShowTrajectories(e.target.checked)}
              />
              Show trajectories
            </label>
          </div>
        </div>
      )}

      {/* Timeline Slider */}
      <div className="timeline">
        <label>Replay Time: {timeSlider.toFixed(1)}s</label>
        <input
          type="range"
          min="0"
          max="10"
          step="0.1"
          value={timeSlider}
          onChange={(e) => setTimeSlider(parseFloat(e.target.value))}
          className="slider"
        />
      </div>

      {/* Active Learning Prompt */}
      {labelPrompt && labelPrompt.ui_config.show_modal && (
        <div className="modal-overlay">
          <div className="modal">
            <h3>Label This Play</h3>
            <p className="flagged-reason">{labelPrompt.flagged_reason}</p>

            <div className="decision-summary">
              <strong>Predicted:</strong> {labelPrompt.decision}
              <br />
              <strong>Confidence:</strong> {labelPrompt.confidence}
            </div>

            <div className="button-group">
              {labelPrompt.ui_config.buttons.map((btn) => (
                <button
                  key={btn.id}
                  className={`outcome-button ${btn.color}`}
                  onClick={() => handleLabelSubmit(btn.label)}
                >
                  {btn.label}
                </button>
              ))}
            </div>

            <div className="tags-section">
              <label>Add Tags:</label>
              <div className="tag-suggestions">
                {labelPrompt.ui_config.tag_suggestions.map((tag) => (
                  <button
                    key={tag}
                    className={`tag-button ${selectedTags.includes(tag) ? 'selected' : ''}`}
                    onClick={() => toggleTag(tag)}
                  >
                    {tag}
                  </button>
                ))}
              </div>

              <div className="custom-tag">
                <input
                  type="text"
                  placeholder="Custom tag..."
                  value={customTag}
                  onChange={(e) => setCustomTag(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && addCustomTag()}
                />
                <button onClick={addCustomTag}>Add</button>
              </div>

              {selectedTags.length > 0 && (
                <div className="selected-tags">
                  <strong>Selected:</strong> {selectedTags.join(', ')}
                </div>
              )}
            </div>

            <button
              className="close-button"
              onClick={() => setLabelPrompt(null)}
            >
              Skip
            </button>
          </div>
        </div>
      )}

      {/* Evidence Panel */}
      <div className="evidence-panel">
        <h4>Evidence</h4>
        <ul>
          <li>WR open in {insight.details.zone} zone</li>
          <li>Low defensive pressure ({(insight.details.opportunity * 100).toFixed(0)}%)</li>
          <li>xT gradient favorable (+{insight.details.xt_improvement}%)</li>
          {counterfactuals && counterfactuals.cvs >= 0.92 && (
            <li>High counterfactual validity ({(counterfactuals.cvs * 100).toFixed(1)}%)</li>
          )}
        </ul>
      </div>
    </div>
  )
}

export default App
