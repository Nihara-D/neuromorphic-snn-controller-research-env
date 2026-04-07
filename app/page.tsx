"use client"

import { useState, useEffect, useRef } from "react"
import { Lock, Shield, FileWarning, AlertTriangle } from "lucide-react"

// Access key stored as SHA-256 hash — never stored in plaintext
const ACCESS_KEY_HASH = "9c5fa49630a3645624d554e710c8437b74ab7fdc8308e7bda6ebf3fe6084a5eb"

async function hashKey(input: string): Promise<string> {
  const encoder = new TextEncoder()
  const data = encoder.encode(input)
  const hashBuffer = await crypto.subtle.digest("SHA-256", data)
  return Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
}

// Neural network animation canvas
function NeuralCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const resize = () => {
      canvas.width = canvas.offsetWidth * 2
      canvas.height = canvas.offsetHeight * 2
      ctx.scale(2, 2)
    }
    resize()
    window.addEventListener("resize", resize)

    // Neural nodes
    interface Node {
      x: number
      y: number
      vx: number
      vy: number
      radius: number
      firing: number
      lastFire: number
    }

    const nodes: Node[] = []
    const nodeCount = 40

    for (let i = 0; i < nodeCount; i++) {
      nodes.push({
        x: Math.random() * canvas.offsetWidth,
        y: Math.random() * canvas.offsetHeight,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        radius: 2 + Math.random() * 2,
        firing: 0,
        lastFire: 0,
      })
    }

    // Spike train visualization
    const spikes: { x: number; y: number; alpha: number; targetX: number; targetY: number }[] = []

    let frame = 0
    const animate = () => {
      frame++
      ctx.fillStyle = "rgba(10, 10, 12, 0.15)"
      ctx.fillRect(0, 0, canvas.offsetWidth, canvas.offsetHeight)

      // Update and draw nodes
      nodes.forEach((node, i) => {
        node.x += node.vx
        node.y += node.vy

        // Bounce off edges
        if (node.x < 0 || node.x > canvas.offsetWidth) node.vx *= -1
        if (node.y < 0 || node.y > canvas.offsetHeight) node.vy *= -1

        // Random firing (Poisson-like)
        if (Math.random() < 0.005 && frame - node.lastFire > 30) {
          node.firing = 1
          node.lastFire = frame

          // Create spike to random connected node
          const target = nodes[Math.floor(Math.random() * nodes.length)]
          if (target !== node) {
            spikes.push({
              x: node.x,
              y: node.y,
              alpha: 1,
              targetX: target.x,
              targetY: target.y,
            })
          }
        }

        node.firing *= 0.92

        // Draw node
        const baseColor = node.firing > 0.1 ? `rgba(200, 60, 60, ${0.4 + node.firing * 0.6})` : "rgba(80, 80, 90, 0.5)"
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius + node.firing * 3, 0, Math.PI * 2)
        ctx.fillStyle = baseColor
        ctx.fill()

        // Draw connections to nearby nodes
        nodes.slice(i + 1).forEach((other) => {
          const dx = other.x - node.x
          const dy = other.y - node.y
          const dist = Math.sqrt(dx * dx + dy * dy)

          if (dist < 100) {
            ctx.beginPath()
            ctx.moveTo(node.x, node.y)
            ctx.lineTo(other.x, other.y)
            ctx.strokeStyle = `rgba(60, 60, 70, ${0.15 * (1 - dist / 100)})`
            ctx.lineWidth = 0.5
            ctx.stroke()
          }
        })
      })

      // Update and draw spikes
      for (let i = spikes.length - 1; i >= 0; i--) {
        const spike = spikes[i]
        const dx = spike.targetX - spike.x
        const dy = spike.targetY - spike.y
        const dist = Math.sqrt(dx * dx + dy * dy)

        if (dist < 5 || spike.alpha < 0.1) {
          spikes.splice(i, 1)
          continue
        }

        spike.x += (dx / dist) * 3
        spike.y += (dy / dist) * 3
        spike.alpha *= 0.96

        ctx.beginPath()
        ctx.arc(spike.x, spike.y, 2, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(200, 80, 80, ${spike.alpha})`
        ctx.fill()
      }

      requestAnimationFrame(animate)
    }

    animate()

    return () => window.removeEventListener("resize", resize)
  }, [])

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
}

// Redacted text effect
function RedactedText({ text, delay = 0 }: { text: string; delay?: number }) {
  const [revealed, setRevealed] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setRevealed(true), delay)
    return () => clearTimeout(timer)
  }, [delay])

  if (!revealed) {
    return <span className="bg-foreground/80 text-transparent select-none">{text}</span>
  }

  return <span>{text}</span>
}

export default function ConfidentialResearchPage() {
  const [accessAttempts, setAccessAttempts] = useState(0)
  const [showWarning, setShowWarning] = useState(false)
  const [showLogin, setShowLogin] = useState(false)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [loginError, setLoginError] = useState("")

  const handleAccessAttempt = () => {
    setShowLogin(true)
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    const inputHash = await hashKey(password)
    if (inputHash === ACCESS_KEY_HASH) {
      setShowLogin(false)
      setLoginError("")
      setShowWarning(false)
      // Access granted — extend here for authenticated view
      alert("Access granted. Welcome.")
    } else {
      setAccessAttempts((prev) => prev + 1)
      setLoginError(`Access denied. Invalid credentials. Attempt ${accessAttempts + 1} has been logged and reported.`)
      setTimeout(() => setLoginError(""), 4000)
    }
  }

  const handleCloseLogin = () => {
    setShowLogin(false)
    setEmail("")
    setPassword("")
    setLoginError("")
  }

  return (
    <main className="relative min-h-screen bg-[#0a0a0c] text-neutral-200 overflow-hidden">
      {/* Neural network background */}
      <NeuralCanvas />

      {/* Scan lines overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.03]"
        style={{
          backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.03) 2px, rgba(255,255,255,0.03) 4px)",
        }}
      />

      {/* Content */}
      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Header */}
        <header className="border-b border-neutral-800/50 backdrop-blur-sm bg-[#0a0a0c]/60">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded bg-red-900/30 border border-red-800/50 flex items-center justify-center">
                <Shield className="w-4 h-4 text-red-500" />
              </div>
              <div>
                <p className="text-xs text-neutral-500 uppercase tracking-wider">Classification Level</p>
                <p className="text-sm font-mono text-red-500">CONFIDENTIAL</p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs text-neutral-500">
              <Lock className="w-3 h-3" />
              <span>Research Division</span>
            </div>
          </div>
        </header>

        {/* Main content */}
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="max-w-2xl w-full">
            {/* Warning banner */}
            <div className="mb-8 p-4 border border-red-900/50 bg-red-950/20 rounded-lg">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-400">Restricted Access</p>
                  <p className="text-xs text-neutral-400 mt-1">
                    This research environment contains unpublished proprietary work. 
                    Unauthorized access, reproduction, or distribution is strictly prohibited.
                  </p>
                </div>
              </div>
            </div>

            {/* Main card */}
            <div className="border border-neutral-800/50 bg-[#0a0a0c]/80 backdrop-blur-sm rounded-lg overflow-hidden">
              <div className="border-b border-neutral-800/50 px-6 py-4">
                <div className="flex items-center gap-2">
                  <FileWarning className="w-4 h-4 text-neutral-500" />
                  <span className="text-xs text-neutral-500 uppercase tracking-wider">Project Status</span>
                </div>
              </div>

              <div className="p-6 space-y-6">
                <div>
                  <h1 className="text-2xl font-medium text-neutral-100 mb-2">
                    <RedactedText text="Neuromorphic Control Systems" delay={500} />
                  </h1>
                  <p className="text-sm text-neutral-500">
                    <RedactedText text="Advanced research in spiking neural networks for robotic applications" delay={800} />
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-neutral-900/50 rounded border border-neutral-800/50">
                    <p className="text-xs text-neutral-500 uppercase tracking-wider mb-1">Status</p>
                    <p className="text-sm font-mono text-amber-500">Active Research</p>
                  </div>
                  <div className="p-4 bg-neutral-900/50 rounded border border-neutral-800/50">
                    <p className="text-xs text-neutral-500 uppercase tracking-wider mb-1">Publication</p>
                    <p className="text-sm font-mono text-red-500">Unpublished</p>
                  </div>
                </div>

                <div className="p-4 bg-neutral-900/30 rounded border border-neutral-800/30">
                  <p className="text-xs text-neutral-500 mb-3">Research Components</p>
                  <div className="space-y-2">
                    {[
                      "Module Alpha",
                      "Module Beta", 
                      "Module Gamma",
                    ].map((item, i) => (
                      <div key={i} className="flex items-center gap-2 text-sm">
                        <div className="w-1.5 h-1.5 rounded-full bg-neutral-600" />
                        <span className="text-neutral-400 bg-neutral-800 px-2 py-0.5 rounded select-none">{item}</span>
                        <span className="ml-auto text-xs font-mono text-red-600">[CLASSIFIED]</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Access button */}
                <button
                  onClick={handleAccessAttempt}
                  className="w-full py-3 px-4 bg-neutral-800/50 hover:bg-neutral-800/70 border border-neutral-700/50 rounded text-sm text-neutral-300 transition-colors flex items-center justify-center gap-2"
                >
                  <Lock className="w-4 h-4" />
                  Request Access
                </button>


              </div>
            </div>

            {/* Footer note */}
            <p className="mt-6 text-center text-xs text-neutral-600">
              This environment is monitored. All access attempts are logged and reviewed.
            </p>
          </div>
        </div>

        {/* Footer */}
        <footer className="border-t border-neutral-800/50 backdrop-blur-sm bg-[#0a0a0c]/60">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between text-xs text-neutral-600">
            <span>© {new Date().getFullYear()} shniharard@gmail.com — All Rights Reserved</span>
            <span className="font-mono">Internal Research Environment</span>
          </div>
        </footer>
      </div>

      {/* Login Modal */}
      {showLogin && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div 
            className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            onClick={handleCloseLogin}
          />
          <div className="relative w-full max-w-md border border-neutral-800 bg-[#0a0a0c] rounded-lg shadow-2xl">
            <div className="border-b border-neutral-800 px-6 py-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Lock className="w-4 h-4 text-red-500" />
                <span className="text-sm font-medium text-neutral-200">Secure Authentication</span>
              </div>
              <button 
                onClick={handleCloseLogin}
                className="text-neutral-500 hover:text-neutral-300 text-xl leading-none"
              >
                x
              </button>
            </div>
            
            <form onSubmit={handleLogin} className="p-6 space-y-4">
              <div className="p-3 bg-amber-950/30 border border-amber-900/50 rounded text-xs text-amber-400">
                This system requires Level 4 security clearance. All login attempts are monitored and recorded.
              </div>
              
              <div>
                <label className="block text-xs text-neutral-500 uppercase tracking-wider mb-2">
                  Institutional Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="researcher@institution.edu"
                  className="w-full px-4 py-3 bg-neutral-900 border border-neutral-800 rounded text-sm text-neutral-200 placeholder:text-neutral-600 focus:outline-none focus:border-neutral-700"
                  required
                />
              </div>
              
              <div>
                <label className="block text-xs text-neutral-500 uppercase tracking-wider mb-2">
                  Access Key
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your access key"
                  className="w-full px-4 py-3 bg-neutral-900 border border-neutral-800 rounded text-sm text-neutral-200 placeholder:text-neutral-600 focus:outline-none focus:border-neutral-700"
                  required
                />
              </div>

              {loginError && (
                <div className="p-3 bg-red-950/50 border border-red-900/50 rounded text-xs text-red-400 animate-in fade-in duration-200">
                  {loginError}
                </div>
              )}
              
              <button
                type="submit"
                className="w-full py-3 px-4 bg-red-900/30 hover:bg-red-900/50 border border-red-800/50 rounded text-sm text-red-300 transition-colors flex items-center justify-center gap-2"
              >
                <Shield className="w-4 h-4" />
                Authenticate
              </button>
              
              <p className="text-xs text-neutral-600 text-center">
                Unauthorized access attempts will be reported to security.
              </p>
            </form>
          </div>
        </div>
      )}
    </main>
  )
}
