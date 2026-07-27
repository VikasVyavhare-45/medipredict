"""
app.py
Main entry point for the MediPredict Streamlit app.
Handles: DB init, session state, login/register gate, and navigation
between Home / disease pages / Dashboard (each of those lives in pages/).

The "not logged in" screen renders the original Index.html marketing
landing page (hero, disease grid, steps, features, CTA, footer) exactly
as designed, with the "Log in" / "Get started" buttons scrolling down
to a native Streamlit login/register form wired into database.py.

Run from the project root:
    streamlit run app.py
"""

import sys
import os
import streamlit as st
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
os.makedirs("database", exist_ok=True)
os.makedirs("reports", exist_ok=True)

from database import (
    init_db,
    create_user,
    verify_login,
    get_user_history,
    find_user_by_username_email,
    reset_password,
)

st.set_page_config(
    page_title="MediPredict",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded" if st.session_state.get("logged_in") else "collapsed",
)

# ---- One-time DB setup ----
init_db()

# ---- Session state defaults ----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None


def safe_html(html_str):
    """Render raw HTML/CSS without going through Streamlit's Markdown parser.
    st.html() (Streamlit >= 1.36) renders HTML directly, so indentation of
    the source string can never accidentally trigger Markdown's "indented
    code block" rule (which is what caused CSS to show up as literal text
    earlier). Falls back to st.markdown for older Streamlit installs."""
    if hasattr(st, "html"):
        st.html(html_str)
    else:
        st.markdown(html_str, unsafe_allow_html=True)

LANDING_CSS = """

  :root{
    color-scheme: light dark;
    --ink: #1E1B3D;
    --ink-soft: #5D5680;
    --ink-faint: #8983AD;
    --forest-deep: #1B1740;
    --forest: #3D2E8C;
    --forest-mid: #5B3FC4;
    --green: #7C3AED;
    --leaf: #A78BFA;
    --lime: #C9B8FA;
    --sage: #F2EFFC;
    --card: #FFFFFF;
    --line: #E1DCF5;
    --gold: #E8B34E;
    --coral: #D9603F;
    --nav-bg: rgba(242,239,252,0.86);
    --shadow-ink: rgba(27,23,64,0.4);
    /* --forest-deep / --forest are TEXT-ink colors (headings, eyebrow,
       trust numbers, disease-strip text) and flip to a light color in
       dark mode below. --panel-deep / --panel are a SEPARATE, constant
       pair used only for the always-dark decorative surfaces (hero card,
       CTA band, gradient buttons) that carry white text in both themes -
       they must never flip, or those surfaces go dark-text-on-dark. */
    --panel-deep: #1B1740;
    --panel: #3D2E8C;
  }
  /* Dark palette - kicks in automatically when the visitor's OS/browser
     is set to dark mode, no toggle needed. Accent colors (green/leaf/
     lime/gold/coral) stay the same across both themes so the brand still
     reads correctly; only the surfaces and text swap for contrast. */
  @media (prefers-color-scheme: dark){
    :root{
      --ink: #EDE9FB;
      --ink-soft: #B3ABD6;
      --ink-faint: #857CAD;
      --forest-deep: #F5F1FF;
      --forest: #C9B8FA;
      --forest-mid: #382A78;
      --green: #9D6FF0;
      --leaf: #B7A2F7;
      --lime: #D6C9FA;
      --sage: #100D22;
      --card: #1B1732;
      --line: #322B58;
      --gold: #E8B34E;
      --coral: #E17A57;
      --nav-bg: rgba(16,13,34,0.86);
      --shadow-ink: rgba(0,0,0,0.55);
    }
  }
  *{box-sizing:border-box; margin:0; padding:0;}
  html{scroll-behavior:smooth;}
  body{
    background:var(--sage);
    color:var(--ink);
    font-family:'Inter', sans-serif;
    -webkit-font-smoothing:antialiased;
    overflow-x:hidden;
  }
  a{color:inherit; text-decoration:none;}

  /* NAVBAR */
  header{
    position:sticky; top:0; z-index:50;
    background:var(--nav-bg);
    backdrop-filter:blur(10px);
    border-bottom:1px solid var(--line);
  }
  .nav-wrap{max-width:1180px; margin:0 auto; padding:16px 40px; display:flex; align-items:center; justify-content:space-between;}
  .brand{font-family:'Fraunces', serif; font-size:21px; font-weight:600; letter-spacing:-0.01em;}
  .brand span{color:var(--green);}
  .nav-links{display:flex; gap:30px; align-items:center; font-size:13.5px; color:var(--ink-soft); font-weight:500;}
  .nav-links a:hover{color:var(--ink);}
  .nav-cta{display:flex; gap:10px; align-items:center;}
  .btn{
    font-family:'Inter'; font-size:13.5px; font-weight:600; padding:10px 18px; border-radius:9px;
    border:none; cursor:pointer; display:inline-flex; align-items:center; gap:8px;
    transition:transform .12s ease, box-shadow .12s ease, background .15s ease;
  }
  .btn-primary{background:linear-gradient(160deg, var(--green), var(--panel-deep)); color:#fff; box-shadow:0 8px 18px -6px rgba(27,23,64,0.4);}
  .btn-primary:hover{transform:translateY(-1px); box-shadow:0 12px 22px -6px rgba(27,23,64,0.48);}
  .btn-ghost{background:transparent; color:var(--ink); border:1px solid var(--line);}
  .btn-ghost:hover{border-color:var(--green);}
  .btn-light{background:#fff; color:var(--panel-deep);}
  .btn-light:hover{transform:translateY(-1px); box-shadow:0 10px 20px rgba(0,0,0,0.18);}

  /* HERO */
  .hero{
    position:relative;
    max-width:1180px; margin:0 auto; padding:76px 40px 60px;
    display:grid;
    grid-template-columns: 1.05fr 1fr;
    align-items:center;
    gap:56px;
  }
  .hero-glow{
    position:absolute; top:-120px; left:-160px; width:520px; height:520px; border-radius:50%;
    background:radial-gradient(circle, rgba(201,184,250,0.22), transparent 70%);
    filter:blur(10px); z-index:0; pointer-events:none;
  }
  .eyebrow{
    display:inline-flex; align-items:center; gap:8px; font-size:11.5px; text-transform:uppercase;
    letter-spacing:.1em; font-weight:700; color:var(--forest-deep); background:var(--card);
    border:1px solid var(--line); padding:6px 14px; border-radius:100px; margin-bottom:20px;
    position:relative; z-index:1;
  }
  .eyebrow::before{content:''; width:6px; height:6px; border-radius:50%; background:var(--green);}
  .hero-copy{position:relative; z-index:1;}
  .hero h1{font-family:'Fraunces', serif; font-size:46px; line-height:1.12; font-weight:600; letter-spacing:-0.015em; margin-bottom:18px;}
  .hero h1 em{font-style:italic; color:var(--forest); font-weight:500;}
  .hero p.lead{font-size:15.5px; color:var(--ink-soft); line-height:1.65; max-width:480px; margin-bottom:30px;}
  .hero-actions{display:flex; gap:12px; align-items:center; margin-bottom:34px;}
  .hero-actions .btn{padding:13px 24px; font-size:14px;}
  .trust-row{display:flex; gap:26px; flex-wrap:wrap;}
  .trust-item{display:flex; flex-direction:column; gap:2px;}
  .trust-item .num{font-family:'IBM Plex Mono', monospace; font-size:20px; font-weight:600; color:var(--forest-deep);}
  .trust-item .lbl{font-size:11.5px; color:var(--ink-soft);}

  /* HERO VISUAL — same 3D brand-panel language as register/profile */
  .hero-visual{
    position:relative; overflow:hidden; border-radius:20px; padding:26px; color:#fff;
    background:
      radial-gradient(500px 400px at 20% 10%, rgba(255,255,255,0.08), transparent 60%),
      linear-gradient(150deg, var(--panel-deep) 0%, var(--panel) 55%, var(--forest-mid) 100%);
    box-shadow: 0 30px 70px -20px rgba(27,23,64,0.5);
    perspective:1200px;
  }
  .noise-ring{position:absolute; width:340px; height:340px; border-radius:50%; border:1px solid rgba(255,255,255,0.08); top:-110px; right:-110px;}
  .noise-ring.two{width:220px; height:220px; border-color:rgba(255,255,255,0.06); top:-60px; right:-60px;}
  .orb{
    position:absolute; z-index:1; border-radius:50%;
    background:
      radial-gradient(circle at 32% 28%, rgba(255,255,255,0.5), rgba(201,184,250,0.14) 40%, transparent 60%),
      radial-gradient(circle at 65% 70%, rgba(58,42,122,0.9), rgba(27,23,64,0.95) 70%);
    box-shadow: inset -14px -14px 32px rgba(0,0,0,0.45), inset 10px 10px 24px rgba(255,255,255,0.12), 0 30px 60px rgba(0,0,0,0.5);
    animation: orbFloat 9s ease-in-out infinite;
  }
  .orb.o1{width:110px; height:110px; bottom:-30px; left:-30px; opacity:0.75;}
  .orb.o2{width:50px; height:50px; bottom:60px; left:40px; opacity:0.6; animation-delay:-3s;}
  @keyframes orbFloat{0%,100%{transform:translateY(0) translateX(0);} 50%{transform:translateY(-14px) translateX(6px);}}
  @media (prefers-reduced-motion: reduce){ .orb{animation:none;} }

  .hv-inner{position:relative; z-index:2; transform-style:preserve-3d; transition:transform .25s ease-out;}
  .hv-top{display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;}
  .hv-tag{font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:#CBBEF5; font-weight:600;}
  .hv-card{
    background:rgba(255,255,255,0.09); border:1px solid rgba(255,255,255,0.16);
    border-radius:14px; padding:18px; margin-bottom:14px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.25);
  }
  .hv-card .disease{font-size:13px; color:#D9D0F5; margin-bottom:8px;}
  .hv-row{display:flex; align-items:center; justify-content:space-between;}
  .hv-score{font-family:'IBM Plex Mono', monospace; font-size:28px; font-weight:600;}
  .hv-badge{font-size:11px; font-weight:600; padding:4px 10px; border-radius:100px; background:rgba(201,184,250,0.2); color:var(--lime); border:1px solid rgba(201,184,250,0.4);}
  .hv-badge.warn{background:rgba(217,96,63,0.2); color:#FFCFBE; border-color:rgba(217,96,63,0.4);}
  .hv-bars{display:flex; gap:5px; align-items:flex-end; height:42px; margin-top:12px;}
  .hv-bars div{flex:1; background:var(--leaf); border-radius:3px; opacity:.85;}

  /* MARQUEE / DISEASE STRIP */
  .disease-strip{border-top:1px solid var(--line); border-bottom:1px solid var(--line); background:var(--card); padding:14px 0; overflow:hidden;}
  .strip-track{display:flex; gap:36px; white-space:nowrap; font-family:'IBM Plex Mono', monospace; font-size:12.5px; color:var(--forest-deep); animation:scroll 28s linear infinite; width:max-content;}
  @keyframes scroll{ from{transform:translateX(0);} to{transform:translateX(-50%);} }
  .strip-track span{opacity:.8;}
  .strip-track span::after{content:'·'; margin-left:36px; opacity:.5;}

  /* SECTIONS */
  section.section{max-width:1180px; margin:0 auto; padding:80px 40px;}
  .section-head{max-width:560px; margin-bottom:44px;}
  .section-head .eyebrow{margin-bottom:14px;}
  .section-head h2{font-family:'Fraunces', serif; font-size:32px; font-weight:600; letter-spacing:-0.01em; margin-bottom:12px;}
  .section-head p{color:var(--ink-soft); font-size:14.5px; line-height:1.6;}

  /* DISEASE GRID — 3D tilt cards */
  .disease-grid{display:grid; grid-template-columns:repeat(5, 1fr); gap:14px;}
  .d-card{
    background:var(--card); border:1px solid var(--line); border-radius:13px; padding:20px 16px;
    transition:border-color .15s ease, transform .12s ease, box-shadow .2s ease;
    cursor:pointer; transform:perspective(800px);
  }
  .d-card:hover{border-color:var(--green);}
  .d-card .d-icon{
    width:36px; height:36px; border-radius:10px; color:#fff;
    background:linear-gradient(155deg, var(--green), var(--panel-deep));
    display:flex; align-items:center; justify-content:center; font-size:16px; margin-bottom:14px;
    box-shadow:0 6px 14px -4px rgba(27,23,64,0.4);
  }
  .d-card .d-name{font-size:13.5px; font-weight:600; margin-bottom:4px;}
  .d-card .d-algo{font-size:10.5px; color:var(--ink-soft); line-height:1.4;}

  /* HOW IT WORKS */
  .steps{display:grid; grid-template-columns:repeat(4,1fr); gap:20px;}
  .step-card{padding:22px 0; border-top:2px solid var(--forest-deep);}
  .step-card .s-num{font-family:'IBM Plex Mono', monospace; font-size:12px; color:var(--green); font-weight:600; margin-bottom:12px;}
  .step-card h3{font-family:'Fraunces', serif; font-size:16.5px; font-weight:600; margin-bottom:8px;}
  .step-card p{font-size:12.5px; color:var(--ink-soft); line-height:1.55;}

  /* FEATURES */
  .feat-grid{display:grid; grid-template-columns:repeat(3, 1fr); gap:16px;}
  .f-card{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:24px; transition:transform .12s ease, box-shadow .2s ease; transform:perspective(800px);}
  .f-card .f-icon{
    width:34px; height:34px; border-radius:9px; display:flex; align-items:center; justify-content:center;
    background:var(--sage); color:var(--forest-deep); font-size:16px; margin-bottom:14px;
  }
  .f-card h3{font-family:'Fraunces', serif; font-size:15.5px; font-weight:600; margin-bottom:8px;}
  .f-card p{font-size:12.5px; color:var(--ink-soft); line-height:1.6;}

  /* HEALTH HISTORY SHOWCASE */
  .history-section{
    position:relative; overflow:hidden;
    max-width:1180px; margin:0 auto 60px; padding:60px 50px;
    background:
      radial-gradient(600px 400px at 85% 15%, rgba(255,255,255,0.07), transparent 60%),
      linear-gradient(150deg, var(--panel-deep) 0%, var(--panel) 60%, var(--forest-mid) 100%);
    border-radius:20px; color:#fff;
    display:grid; grid-template-columns:1fr 1fr; gap:40px; align-items:center;
  }
  .history-section h2{font-family:'Fraunces', serif; font-size:30px; font-weight:600; line-height:1.2; margin-bottom:14px; position:relative; z-index:1;}
  .history-section p{font-size:14px; color:#D9D0F5; line-height:1.65; max-width:420px; position:relative; z-index:1;}
  .history-cards{position:relative; height:230px;}
  .hist-card{
    position:absolute; width:280px; background:rgba(255,255,255,0.08);
    border:1px solid rgba(255,255,255,0.16); border-radius:14px; padding:18px 20px;
    box-shadow:0 20px 40px rgba(0,0,0,0.3); backdrop-filter:blur(2px);
  }
  .hist-card .hc-label{font-size:10.5px; text-transform:uppercase; letter-spacing:.08em; color:#B3A3EE; font-weight:600; margin-bottom:8px;}
  .hist-card .hc-val{font-family:'IBM Plex Mono', monospace; font-size:22px; font-weight:600; margin-bottom:4px;}
  .hist-card .hc-sub{font-size:11.5px; color:#CBBEF5;}
  .hist-card.c1{top:0; left:10px; transform:rotate(-7deg); z-index:1;}
  .hist-card.c2{top:55px; left:65px; transform:rotate(-2deg); z-index:2;}
  .hist-card.c3{top:112px; left:120px; transform:rotate(3deg); z-index:3;}
  @media (max-width: 980px){
    .history-section{grid-template-columns:1fr; padding:40px 30px;}
    .history-cards{margin-top:20px;}
  }

  /* CTA BAND */
  .cta-band{
    position:relative; overflow:hidden;
    max-width:1180px; margin:0 auto 80px; padding:50px 50px;
    background:
      radial-gradient(500px 400px at 85% 20%, rgba(255,255,255,0.08), transparent 60%),
      linear-gradient(135deg, var(--panel-deep), var(--forest-mid));
    border-radius:20px; color:#fff;
    display:flex; align-items:center; justify-content:space-between; gap:30px;
  }
  .cta-band .orb{width:150px; height:150px; bottom:-50px; right:60px; opacity:0.5;}
  .cta-band h2{font-family:'Fraunces', serif; font-size:26px; font-weight:600; margin-bottom:8px; position:relative; z-index:1;}
  .cta-band p{font-size:13.5px; color:#D9D0F5; max-width:420px; position:relative; z-index:1;}
  .cta-band .btn-light{padding:13px 24px; font-size:13.5px; white-space:nowrap; position:relative; z-index:1;}

  footer.site-foot{border-top:1px solid var(--line); padding:34px 40px; text-align:center; font-size:11.5px; color:var(--ink-soft);}

  @media (max-width: 980px){
    .hero{padding-top:50px; grid-template-columns:1fr; gap:36px;}
    .hero-visual{max-width:440px; margin:0 auto;}
    .nav-links{display:none;}
    .disease-grid{grid-template-columns:repeat(2,1fr);}
    .steps, .feat-grid{grid-template-columns:1fr 1fr;}
    .cta-band{flex-direction:column; text-align:center;}
  }

  /* PHONE — tighter spacing + smaller type so nothing overflows or
     forces horizontal scroll on real phone-width screens. */
  @media (max-width: 640px){
    .nav-wrap{padding:12px 16px;}
    .brand{font-size:18px;}
    .btn{padding:9px 13px; font-size:12.5px;}
    .hero{padding:36px 18px 40px; gap:28px;}
    .hero h1{font-size:30px;}
    .hero p.lead{font-size:14px; max-width:100%;}
    .hero-actions{flex-wrap:wrap; gap:10px;}
    .hero-actions .btn{padding:12px 18px; font-size:13px;}
    .trust-row{gap:16px;}
    .hero-visual{padding:18px; max-width:100%;}
    .hv-score{font-size:22px;}
    section.section{padding:44px 18px;}
    .section-head h2{font-size:24px;}
    .disease-grid{grid-template-columns:repeat(2,1fr); gap:10px;}
    .d-card{padding:16px 12px;}
    .steps, .feat-grid{grid-template-columns:1fr;}
    .history-section{padding:30px 20px; margin:0 16px 40px;}
    .history-section h2{font-size:22px;}
    .history-cards{height:190px;}
    .hist-card{width:78%;}
    .cta-band{padding:34px 22px; margin:0 16px 50px;}
    .cta-band h2{font-size:21px;}
    footer.site-foot{padding:24px 18px;}
  }

"""
LANDING_BODY = """


<header>
  <div class="nav-wrap">
    <div class="brand">Medi<span>Predict</span></div>
    <div class="nav-links">
      <a href="#diseases">Conditions</a>
      <a href="#how">How it works</a>
      <a href="#features">Features</a>
    </div>
    <div class="nav-cta">
      <a href="?auth=login" class="btn btn-ghost">Log in</a>
      <a href="?auth=register" class="btn btn-primary">Get started →</a>
    </div>
  </div>
</header>

<section class="hero">
  <div class="hero-glow"></div>
  <div class="hero-copy">
    <div class="eyebrow">10 conditions · ML-powered</div>
    <h1>Know your risk, <em>before</em><br>it becomes a diagnosis.</h1>
    <p class="lead">Enter your medical values and get instant, explainable predictions across 10 diseases — with confidence scores, personalised suggestions, and an AI you can ask follow-up questions.</p>
    <div class="hero-actions">
      <a href="?auth=register" class="btn btn-primary">Create free account</a>
      <a href="#diseases" class="btn btn-ghost">See supported conditions</a>
    </div>
    <div class="trust-row">
      <div class="trust-item"><div class="num">10</div><div class="lbl">Diseases covered</div></div>
      <div class="trust-item"><div class="num">91%</div><div class="lbl">Best-model accuracy</div></div>
      <div class="trust-item"><div class="num">3</div><div class="lbl">Algorithms compared per model</div></div>
    </div>
  </div>
  <div class="hero-visual" id="heroVisual">
    <div class="noise-ring"></div>
    <div class="noise-ring two"></div>
    <div class="orb o1"></div>
    <div class="orb o2"></div>
    <div class="hv-inner" id="hvInner">
      <div class="hv-top">
        <div class="hv-tag">Live prediction</div>
      </div>
      <div class="hv-card">
        <div class="disease">Diabetes risk assessment</div>
        <div class="hv-row">
          <div class="hv-score">18%</div>
          <div class="hv-badge">Low risk</div>
        </div>
        <div class="hv-bars">
          <div style="height:38%"></div>
          <div style="height:62%"></div>
          <div style="height:44%"></div>
          <div style="height:80%"></div>
          <div style="height:52%"></div>
          <div style="height:70%"></div>
        </div>
      </div>
      <div class="hv-card">
        <div class="disease">Heart disease risk</div>
        <div class="hv-row">
          <div class="hv-score">76%</div>
          <div class="hv-badge warn">High risk</div>
        </div>
      </div>
    </div>
  </div>
</section>

<div class="disease-strip">
  <div class="strip-track">
    <span>Diabetes</span><span>Heart Disease</span><span>Parkinson's</span><span>Liver Disease</span>
    <span>Kidney Disease</span><span>Breast Cancer</span><span>Stroke</span><span>Hepatitis</span>
    <span>Thyroid</span><span>Lung Cancer</span>
    <span>Diabetes</span><span>Heart Disease</span><span>Parkinson's</span><span>Liver Disease</span>
    <span>Kidney Disease</span><span>Breast Cancer</span><span>Stroke</span><span>Hepatitis</span>
    <span>Thyroid</span><span>Lung Cancer</span>
  </div>
</div>

<section class="section" id="diseases">
  <div class="section-head">
    <div class="eyebrow">Supported conditions</div>
    <h2>One platform, ten trained models</h2>
    <p>Each condition runs on its own set of models, benchmarked against each other so you always see the best-performing algorithm.</p>
  </div>
  <div class="disease-grid">
    <div class="d-card tilt-sm"><div class="d-icon">♡</div><div class="d-name">Heart Disease</div><div class="d-algo">Logistic Regression · Random Forest · KNN</div></div>
    <div class="d-card tilt-sm"><div class="d-icon">◐</div><div class="d-name">Diabetes</div><div class="d-algo">Logistic Regression · Random Forest · SVM</div></div>
    <div class="d-card tilt-sm"><div class="d-icon">✎</div><div class="d-name">Parkinson's</div><div class="d-algo">SVM · Random Forest · XGBoost</div></div>
    <div class="d-card tilt-sm"><div class="d-icon">◆</div><div class="d-name">Liver Disease</div><div class="d-algo">Logistic Regression · RF · Decision Tree</div></div>
    <div class="d-card tilt-sm"><div class="d-icon">◔</div><div class="d-name">Kidney Disease</div><div class="d-algo">Random Forest · SVM · Log. Regression</div></div>
    <div class="d-card tilt-sm"><div class="d-icon">✦</div><div class="d-name">Breast Cancer</div><div class="d-algo">SVM · Log. Regression · Random Forest</div></div>
    <div class="d-card tilt-sm"><div class="d-icon">⌁</div><div class="d-name">Stroke</div><div class="d-algo">Random Forest · XGBoost · Log. Regr.</div></div>
    <div class="d-card tilt-sm"><div class="d-icon">◑</div><div class="d-name">Hepatitis</div><div class="d-algo">Decision Tree · RF · Log. Regression</div></div>
    <div class="d-card tilt-sm"><div class="d-icon">⬡</div><div class="d-name">Thyroid</div><div class="d-algo">Random Forest · SVM · KNN</div></div>
    <div class="d-card tilt-sm"><div class="d-icon">◈</div><div class="d-name">Lung Cancer</div><div class="d-algo">Log. Regression · RF · Naive Bayes</div></div>
  </div>
</section>

<section class="section" id="how" style="padding-top:0;">
  <div class="section-head">
    <div class="eyebrow">Process</div>
    <h2>From values to insight, in four steps</h2>
  </div>
  <div class="steps">
    <div class="step-card">
      <div class="s-num">01</div>
      <h3>Choose a condition</h3>
      <p>Pick from 10 supported diseases based on what you'd like to check.</p>
    </div>
    <div class="step-card">
      <div class="s-num">02</div>
      <h3>Enter your values</h3>
      <p>Fill in vitals and lab results from your latest checkup — takes under two minutes.</p>
    </div>
    <div class="step-card">
      <div class="s-num">03</div>
      <h3>Get your prediction</h3>
      <p>See a confidence score, the factors driving it, and how your values compare to normal ranges.</p>
    </div>
    <div class="step-card">
      <div class="s-num">04</div>
      <h3>Ask, save, and track</h3>
      <p>Chat with the AI about your result, download a PDF report, and track trends over time.</p>
    </div>
  </div>
</section>

<section class="section" id="features" style="padding-top:0;">
  <div class="section-head">
    <div class="eyebrow">What you get</div>
    <h2>Built for clarity, not just a yes or no</h2>
  </div>
  <div class="feat-grid">
    <div class="f-card tilt-sm"><div class="f-icon">◎</div><h3>Explainable results</h3><p>SHAP-based feature importance shows exactly which values are driving your risk score.</p></div>
    <div class="f-card tilt-sm"><div class="f-icon">▤</div><h3>Model comparison</h3><p>See accuracy across every algorithm tested for your condition, not just one black box.</p></div>
    <div class="f-card tilt-sm"><div class="f-icon">↓</div><h3>PDF reports</h3><p>Download a shareable report of your result to bring to your doctor.</p></div>
    <div class="f-card tilt-sm"><div class="f-icon">✉</div><h3>Risk alerts</h3><p>Get an email notification automatically when a result comes back high-risk.</p></div>
    <div class="f-card tilt-sm"><div class="f-icon">◷</div><h3>History & trends</h3><p>Every past prediction is saved so you can track how your risk changes over time.</p></div>
    <div class="f-card tilt-sm"><div class="f-icon">✦</div><h3>Ask AI</h3><p>Ask free-text questions about your specific result and get answers grounded in your data.</p></div>
  </div>
</section>

<div class="history-section">
  <div>
    <h2>Your health history, right where you left it.</h2>
    <p>Log in to see your saved predictions, track your risk trends, and pick up your last conversation with the AI.</p>
    <a href="?auth=login" class="btn btn-light" style="margin-top:22px; display:inline-flex;">Log in to view your history →</a>
  </div>
  <div class="history-cards">
    <div class="hist-card c1">
      <div class="hc-label">Blood Pressure</div>
      <div class="hc-val">118 / 76</div>
      <div class="hc-sub">Within normal range</div>
    </div>
    <div class="hist-card c2">
      <div class="hc-label">Glucose</div>
      <div class="hc-val">96 mg/dL</div>
      <div class="hc-sub">Slightly elevated</div>
    </div>
    <div class="hist-card c3">
      <div class="hc-label">Cardiac Risk</div>
      <div class="hc-val">Low</div>
      <div class="hc-sub">Down 3% since March</div>
    </div>
  </div>
</div>

<div class="cta-band">
  <div class="orb"></div>
  <div>
    <h2>Check your risk in under 2 minutes</h2>
    <p>Free to create an account. No credit card, no waiting room.</p>
  </div>
  <a href="?auth=register" class="btn btn-light">Create free account →</a>
</div>

<footer class="site-foot">
  MediPredict — Final Year Project · Predictions are ML-based estimates, not clinical diagnoses. Always consult a licensed doctor.
</footer>


"""
STREAMLIT_THEME_CSS = """
<style>
/* ---- Restyle native Streamlit widgets to match the MediPredict forest theme ---- */
.stApp { background-color: var(--sage); }
h1, h2, h3 { font-family: 'Fraunces', serif; color: var(--forest-deep); }
body, p, div, span, label { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background-color: var(--card);
    border-right: 1px solid var(--line);
}

.stButton>button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(160deg, var(--green), var(--panel-deep));
    color: #FFFFFF;
    border-radius: 9px;
    border: none;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    box-shadow: 0 8px 18px -6px var(--shadow-ink);
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
}
.stButton>button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(160deg, var(--forest-mid), var(--panel-deep));
    box-shadow: 0 14px 26px -6px var(--shadow-ink);
    transform: translateY(-2px);
}
.stButton>button:active,
div[data-testid="stFormSubmitButton"] button:active {
    transform: translateY(0px);
    box-shadow: 0 6px 14px -6px var(--shadow-ink);
}
/* st.form_submit_button() is a different component from st.button() -
   the rule above alone doesn't always win against Streamlit's own
   dark-mode default text color, which is how "Log in" / "Create free
   account" turned near-invisible (dark text on the dark button) for
   anyone with system dark mode on. Force every nested text node too. */
div[data-testid="stFormSubmitButton"] button,
div[data-testid="stFormSubmitButton"] button * {
    color: #FFFFFF !important;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    color: var(--ink-soft);
}
.stTabs [aria-selected="true"] {
    color: var(--forest-deep) !important;
    border-bottom-color: var(--green) !important;
}

div[data-testid="stForm"], .auth-card {
    background:
        radial-gradient(400px 200px at 15% 0%, rgba(124,58,237,0.06), transparent 60%),
        var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 30px;
    box-shadow: 0 30px 60px -20px var(--shadow-ink), 0 2px 8px rgba(27,23,64,0.06);
    transition: box-shadow .2s ease;
}
div[data-testid="stForm"]:hover, .auth-card:hover {
    box-shadow: 0 36px 70px -18px var(--shadow-ink), 0 2px 8px rgba(27,23,64,0.08);
}

.stTextInput input {
    border-radius: 8px !important;
    border: 1px solid var(--line) !important;
    background: var(--card) !important;
    color: var(--ink) !important;
}

/* Password field's show/hide eye icon renders as its own button
   alongside the input, not matched by the rule above. */
div[data-testid="stTextInput"] button {
    background: var(--card) !important;
    border: none !important;
    color: var(--ink-soft) !important;
}
div[data-testid="stTextInput"] button svg {
    fill: var(--ink-soft) !important;
    color: var(--ink-soft) !important;
}

.stAlert { border-radius: 10px; }
/* st.info() / st.warning() / st.error() text - not covered by the
   widget-label rule further below, and left unset it inherits
   Streamlit's own internal theme color instead of ours. */
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlertContentWarning"] p,
div[data-testid="stAlertContentInfo"] p,
div[data-testid="stAlertContentError"] p,
div[data-testid="stAlertContentSuccess"] p {
    color: var(--ink) !important;
}

/* st.dialog() (Log in / Create account popup) renders as its own
   overlay and wasn't fully reached by the variables above by default -
   pin it to the same light/dark palette as the rest of the app so its
   title/inputs always read correctly against its own card background. */
div[data-testid="stDialog"] {
    color-scheme: light dark;
}
div[data-testid="stDialog"] > div {
    background: var(--card) !important;
}
div[data-testid="stDialog"] h1,
div[data-testid="stDialog"] h2,
div[data-testid="stDialog"] h3,
div[data-testid="stDialog"] p,
div[data-testid="stDialog"] label,
div[data-testid="stDialog"] span {
    color: var(--ink) !important;
}
div[data-testid="stDialog"] .stTextInput input {
    background: var(--card) !important;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
    border: 1px solid var(--line) !important;
}
div[data-testid="stDialog"] .stTextInput input::placeholder {
    color: var(--ink-faint) !important;
}

/* Condition tabs - one native st.tabs() per clinical category, each tab
   holding a simple full-width st.button() per disease (icon + name),
   with the algorithm list as a plain caption underneath. Buttons are
   native Streamlit interactive elements - a click is never ambiguous
   the way an href/URL-based navigation or a CSS overlay trick can be. */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { padding: 10px 18px; }

div[class*="st-key-dtab-"] button {
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 14.5px !important;
    font-weight: 600 !important;
}
div[class*="st-key-dtab-"] { margin-bottom: 2px; }

/* "Your recent history" rows, styled as small chips instead of plain bullets */
.history-row {
    display: flex; align-items: center; gap: 12px;
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    padding: 12px 16px; margin-bottom: 8px;
}
.history-row .h-dot { font-size: 12px; }
.history-row .h-name { font-weight: 700; color: var(--ink); min-width: 130px; }
.history-row .h-risk { color: var(--ink-soft); font-size: 13.5px; }
.history-row .h-date { margin-left: auto; color: var(--ink-faint); font-size: 12px; font-family: 'IBM Plex Mono', monospace; }

/* ---- Dashboard overview: stat cards row ---- */
.stat-row { display: flex; gap: 14px; flex-wrap: wrap; margin: 4px 0 22px; }
.stat-card {
    flex: 1 1 190px; background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 18px 20px; position: relative; overflow: hidden;
    box-shadow: 0 2px 8px rgba(27,23,64,0.04);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stat-card:hover { transform: translateY(-2px); box-shadow: 0 10px 22px -8px var(--shadow-ink); }
.stat-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.stat-card.total::before { background: linear-gradient(90deg, var(--green), var(--panel-deep)); }
.stat-card.high::before  { background: linear-gradient(90deg, var(--coral), #B84A2C); }
.stat-card.low::before   { background: linear-gradient(90deg, var(--leaf), var(--green)); }
.stat-card.cond::before  { background: linear-gradient(90deg, var(--gold), #C98A1E); }
.stat-card .stat-num { font-family: 'IBM Plex Mono', monospace; font-size: 28px; font-weight: 600; color: var(--ink); line-height: 1; margin-bottom: 6px; }
.stat-card .stat-lbl { font-size: 12px; color: var(--ink-soft); }

/* ---- Home page header + empty-state block (used inline via safe_html
   in home_page(), kept here so they follow the same theme variables) ---- */
.brand-title { font-family:'Fraunces',serif; font-size:34px; font-weight:600; letter-spacing:-0.01em; color:var(--ink); margin-bottom:2px; }
.brand-title span { color:var(--green); }
.logged-in-as { color:var(--ink-soft); font-size:13.5px; }
.logged-in-as strong { color:var(--ink); }
.home-lead { color:var(--ink-soft); font-size:15px; line-height:1.6; max-width:640px; }
.empty-history { background:var(--card); border:1px dashed var(--line); border-radius:14px; padding:28px; text-align:center; color:var(--ink-soft); font-size:13.5px; margin-bottom:8px; }

/* ---- Force readable text on Streamlit's OWN native widgets ---- */
/* The color-scheme:light rule above stops native form controls (inputs,
   checkboxes) from flipping to a dark browser theme, but it does NOT stop
   Streamlit's own internal dark-theme CSS variables from making plain
   widget labels/captions render near-white when the visitor's OS/browser
   is dark mode. This ONLY targets genuinely-native Streamlit widgets
   (number_input/selectbox/text_input labels, st.caption, st.metric) -
   it deliberately does NOT touch [data-testid="stMarkdownContainer"] or
   .stMarkdown, because ALL of the custom hero/disease-card/history/CTA
   HTML sections above are also delivered via st.markdown() and already
   set their own (often white-on-purple) colors inline - a blanket rule
   there would fight those instead of fixing anything. */
[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
.stSelectbox label, .stNumberInput label, .stTextInput label,
.stCheckbox label, .stRadio label {
    color: var(--ink) !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--ink-soft) !important;
}
.auth-switch-caption {
    font-size: 13.5px;
    color: var(--ink-soft);
    line-height: 1.5;
    margin-top: 2px;
}
.auth-switch-caption a {
    color: var(--green);
    font-weight: 700;
    text-decoration: underline;
    text-underline-offset: 2px;
    cursor: pointer;
}
.auth-switch-caption a:hover {
    color: var(--panel-deep);
}
</style>
"""

def apply_theme():
    # IMPORTANT: none of these lines can start with leading whitespace.
    # st.markdown() runs its input through a Markdown parser first, and
    # Markdown treats any line indented by 4+ spaces as a code block —
    # which renders the raw CSS as visible text instead of applying it
    # as a stylesheet. Keep this fully left-aligned (no indentation).
    # Always-on: hide Streamlit's own Deploy button, hamburger menu, footer
    # badge, the auto-generated sidebar page list, AND the sidebar itself
    # (the whole left column — "Back to Home" link, "Logged in as ..." text,
    # collapse arrow). This app has its own nav baked into each page's main
    # content instead, so the default Streamlit sidebar is just dead space.
    always_css = (
        '<style>\n'
        ':root, .stApp { color-scheme: light dark; }\n'
        '#MainMenu { visibility: hidden; }\n'
        '.stAppDeployButton { display: none !important; }\n'
        'div[data-testid="stToolbarActions"] { display: none !important; }\n'
        'footer { visibility: hidden; }\n'
        '[data-testid="stSidebarNav"] { display: none !important; }\n'
        'header[data-testid="stHeader"] { display: none !important; }\n'
        'section[data-testid="stSidebar"] { display: none !important; }\n'
        'div[data-testid="stSidebarCollapsedControl"] { display: none !important; }\n'
        '.block-container { padding-top: 2rem !important; max-width: 100% !important; }\n'
        '</style>\n'
    )
    chrome_css = ""
    if not st.session_state.logged_in:
        # Landing/login page: on top of the always-on rules above, also
        # zero out the top padding so the hero sits flush under the navbar.
        chrome_css = (
            '<style>\n'
            '.block-container { padding-top: 0 !important; }\n'
            '</style>\n'
        )
    theme_html = (
        '<meta name="color-scheme" content="light dark">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">\n'
        f'<style>{LANDING_CSS}</style>\n'
        f'{STREAMLIT_THEME_CSS}\n'
        f'{always_css}\n'
        f'{chrome_css}'
    )
    safe_html(theme_html)


def hero_tilt_script():
    """Restores the pointer-following 3D tilt effect from the original
    Index.html (hero visual card + disease/feature cards) on hover, for
    devices with a real mouse. Runs in a tiny invisible components.html
    iframe and reaches into the parent page's DOM (same-origin, so this
    is allowed) to attach the listeners — Streamlit's own st.html()/
    st.markdown() never execute <script> tags for security, so a plain
    <script> inside the landing HTML would silently do nothing."""
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            const supportsHoverTilt = window.parent.matchMedia(
                '(hover: hover) and (pointer: fine)'
            ).matches;
            if (!supportsHoverTilt) return;

            function attach() {
                const heroVisual = doc.getElementById('heroVisual');
                const hvInner = doc.getElementById('hvInner');
                if (heroVisual && hvInner && !heroVisual.dataset.tiltBound) {
                    heroVisual.dataset.tiltBound = '1';
                    heroVisual.addEventListener('mousemove', (e) => {
                        const r = heroVisual.getBoundingClientRect();
                        const px = (e.clientX - r.left) / r.width - 0.5;
                        const py = (e.clientY - r.top) / r.height - 0.5;
                        hvInner.style.transform =
                            `rotateX(${-py * 8}deg) rotateY(${px * 8}deg)`;
                    });
                    heroVisual.addEventListener('mouseleave', () => {
                        hvInner.style.transform = 'rotateX(0deg) rotateY(0deg)';
                    });
                }

                doc.querySelectorAll('.tilt-sm').forEach(card => {
                    if (card.dataset.tiltBound) return;
                    card.dataset.tiltBound = '1';
                    card.addEventListener('mousemove', (e) => {
                        const r = card.getBoundingClientRect();
                        const px = (e.clientX - r.left) / r.width - 0.5;
                        const py = (e.clientY - r.top) / r.height - 0.5;
                        card.style.transform =
                            `perspective(800px) rotateX(${-py * 6}deg) rotateY(${px * 6}deg) translateY(-3px)`;
                        card.style.boxShadow =
                            `${-px * 14}px ${12 - py * 12}px 30px rgba(27,23,64,0.12)`;
                    });
                    card.addEventListener('mouseleave', () => {
                        card.style.transform =
                            'perspective(800px) rotateX(0deg) rotateY(0deg) translateY(0px)';
                        card.style.boxShadow = 'none';
                    });
                });
            }

            attach();
            // Re-attach on Streamlit reruns / DOM updates
            new MutationObserver(attach).observe(doc.body, {
                childList: true, subtree: true
            });
        })();
        </script>
        """,
        height=0,
    )


def landing_page():
    """Renders the original Index.html marketing page untouched (hero, disease
    grid, steps, features, CTA, footer). Nav buttons scroll down to #auth-section."""
    safe_html(LANDING_BODY)
    hero_tilt_script()


def login_form():
    prefill_user = st.session_state.pop("prefill_username", "")
    with st.form("login_form_native"):
        username = st.text_input("Username", value=prefill_user, key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Log in →")

    if submitted:
        user = verify_login(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.user = user
            st.success(f"Welcome back, {user['username']}! Redirecting…")
            st.rerun()
        else:
            st.error("Invalid username or password. Please try again.")

    # "Forgot password?" -> closes this dialog and reopens the dedicated
    # Forgot Password dialog on the next rerun.
    if st.button("Forgot password?", key="forgot_password_link"):
        st.session_state.open_forgot_dialog = True
        st.rerun()


def forgot_password_form():
    st.caption(
        "Enter the username and email you registered with, then choose a new password."
    )
    with st.form("forgot_password_form_native"):
        username = st.text_input("Username", key="forgot_username")
        email = st.text_input("Email", key="forgot_email")
        new_password = st.text_input("New Password", type="password", key="forgot_new_password")
        confirm = st.text_input("Confirm New Password", type="password", key="forgot_confirm")
        submitted = st.form_submit_button("Reset password →")

    if submitted:
        if not username or not email or not new_password:
            st.error("Please fill in all fields.")
        elif new_password != confirm:
            st.error("Passwords do not match.")
        else:
            user = find_user_by_username_email(username, email)
            if not user:
                st.error("No account matches that username and email.")
            else:
                reset_password(username, new_password)
                st.success("Password reset successfully! Redirecting you to Login…")
                st.session_state.prefill_username = username
                st.session_state.open_login_dialog = True
                st.rerun()


def register_form():
    with st.form("register_form_native"):
        username = st.text_input("Username", key="reg_username")
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_password")
        confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
        submitted = st.form_submit_button("Create free account →")

    if submitted:
        if not username or not email or not password:
            st.error("Please fill in all fields.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            ok, msg = create_user(username, email, password)
            if ok:
                # Successful registration -> close this dialog, then reopen
                # as the Login dialog with the username already filled in.
                st.success(msg + " Redirecting you to Login…")
                st.session_state.prefill_username = username
                st.session_state.open_login_dialog = True
                st.rerun()
            else:
                # Failed registration -> show error, stay on the Register page
                st.error(msg)


@st.dialog("Log in to MediPredict", width="small")
def login_dialog():
    login_form()
    safe_html(
        '<div class="auth-switch-caption">New here? Close this and hit '
        '<a href="?auth=register" target="_self">Get started</a> to create an account.</div>'
    )


@st.dialog("Create your free account", width="large")
def register_dialog():
    register_form()
    safe_html(
        '<div class="auth-switch-caption">Already have an account? Close this and hit '
        '<a href="?auth=login" target="_self">Log in</a> instead.</div>'
    )


@st.dialog("Reset your password", width="small")
def forgot_password_dialog():
    forgot_password_form()


def auth_section():
    """Opens the Login/Register card as a modal popup on top of the landing
    page. Triggered either by the nav/hero links (?auth=login / ?auth=register
    in the URL) or automatically right after a successful registration."""

    # After a successful registration, jump straight into the Login dialog
    # with the username already filled in.
    if st.session_state.pop("open_login_dialog", False):
        login_dialog()
        return

    # Triggered by the "Forgot password?" link inside the Login dialog.
    if st.session_state.pop("open_forgot_dialog", False):
        forgot_password_dialog()
        return

    auth = st.query_params.get("auth")
    if auth in ("login", "register"):
        # Clear the query param immediately so refreshing the page, or any
        # later rerun, doesn't keep popping the dialog back open.
        st.query_params.pop("auth", None)
        if auth == "login":
            login_dialog()
        else:
            register_dialog()


def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()


def sidebar_nav():
    # The sidebar itself is now hidden globally via CSS in apply_theme(),
    # so nothing needs to be written into st.sidebar here anymore — kept
    # as a no-op stub in case the sidebar is ever brought back later.
    pass


def home_page():
    top_left, top_right = st.columns([5, 1])
    with top_left:
        safe_html(
            '<div class="brand-title">Medi<span>Predict</span></div>'
            f'<div class="logged-in-as">Logged in as '
            f'<strong>{st.session_state.user["username"]}</strong></div>'
        )
    with top_right:
        st.write("")
        if st.button("Logout", use_container_width=True):
            logout()

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    safe_html(
        '<p class="home-lead">'
        'AI-assisted risk screening for 10 diseases, with explainable predictions, '
        'PDF reports, and an AI health chat.</p>'
    )

    # Fetch history once - reused below for both the overview stats/chart
    # and the "recent history" list, instead of querying the DB twice.
    history_all = get_user_history(st.session_state.user["id"]) or []
    total_count = len(history_all)
    high_count = sum(1 for h in history_all if h["prediction"] == 1)
    low_count = total_count - high_count
    conditions_covered = len({h["disease"] for h in history_all})

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    safe_html('<div class="eyebrow">Your activity</div>')
    st.markdown("### Your overview")

    safe_html(
        '<div class="stat-row">'
        f'<div class="stat-card total"><div class="stat-num">{total_count}</div>'
        '<div class="stat-lbl">Total screenings</div></div>'
        f'<div class="stat-card high"><div class="stat-num">{high_count}</div>'
        '<div class="stat-lbl">High-risk flags</div></div>'
        f'<div class="stat-card low"><div class="stat-num">{low_count}</div>'
        '<div class="stat-lbl">Low-risk results</div></div>'
        f'<div class="stat-card cond"><div class="stat-num">{conditions_covered}/10</div>'
        '<div class="stat-lbl">Conditions covered</div></div>'
        '</div>'
    )

    if history_all:
        counts = {}
        for h in history_all:
            counts[h["disease"]] = counts.get(h["disease"], 0) + 1
        chart_df = pd.DataFrame({"Screenings": counts})
        st.bar_chart(chart_df, color="#7C3AED", height=220)
    else:
        safe_html(
            '<div class="empty-history">'
            'No screenings yet — your activity chart will show up here after your first prediction.'
            '</div>'
        )

    disease_categories = [
        ("Cardiometabolic", "cardio", [
            ("Diabetes", "pages/1_Diabetes.py", "◐", "Logistic Regression · Random Forest · SVM"),
            ("Heart Disease", "pages/2_Heart_Disease.py", "♡", "Logistic Regression · Random Forest · KNN"),
            ("Stroke", "pages/7_Stroke.py", "⌁", "Random Forest · XGBoost · Log. Regression"),
        ]),
        ("Organ Function", "organ", [
            ("Liver Disease", "pages/4_Liver_Disease.py", "◆", "Logistic Regression · RF · Decision Tree"),
            ("Kidney Disease", "pages/5_Kidney_Disease.py", "◔", "Random Forest · SVM · Log. Regression"),
            ("Hepatitis", "pages/8_Hepatitis.py", "◑", "Decision Tree · RF · Log. Regression"),
            ("Thyroid", "pages/9_Thyroid.py", "⬡", "Random Forest · SVM · KNN"),
        ]),
        ("Neurological", "neuro", [
            ("Parkinson's", "pages/3_Parkinsons.py", "✎", "SVM · Random Forest · XGBoost"),
        ]),
        ("Oncology", "onco", [
            ("Breast Cancer", "pages/6_Breast_Cancer.py", "✦", "SVM · Log. Regression · Random Forest"),
            ("Lung Cancer", "pages/10_Lung_Cancer.py", "◈", "Log. Regression · RF · Naive Bayes"),
        ]),
    ]
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    safe_html('<div class="eyebrow">10 conditions · ML-powered</div>')
    st.markdown("### Choose a condition to screen")
    st.caption("Pick a clinical panel below, then click a condition to open its screening.")

    cat_tabs = st.tabs([f"{cat_name} ({len(items)})" for cat_name, cat_slug, items in disease_categories])
    for (cat_name, cat_slug, items), tab in zip(disease_categories, cat_tabs):
        with tab:
            for i, (name, page_path, icon, info) in enumerate(items):
                # st.switch_page() is Streamlit's own, official
                # programmatic-navigation API - it's a plain Python
                # call after a real widget click, not a browser URL or
                # any CSS trick, so there is nothing left that can
                # silently fail the way earlier HTML/CSS-based
                # click-through attempts did.
                if st.button(f"{icon}  {name}", key=f"dtab-{cat_slug}-{i}", use_container_width=True):
                    st.switch_page(page_path)
                st.caption(info)

    st.info(
        "⚠️ This is general information, please consult a doctor. "
        "MediPredict does not replace professional medical diagnosis."
    )

    st.markdown("### Your recent history")
    if history_all:
        rows = ""
        for h in history_all[:5]:
            is_high = h["prediction"] == 1
            dot = "🔴" if is_high else "🟢"
            risk_label = "High risk" if is_high else "Low risk"
            rows += (
                '<div class="history-row">'
                f'<span class="h-dot">{dot}</span>'
                f'<span class="h-name">{h["disease"]}</span>'
                f'<span class="h-risk">{risk_label} · {h["confidence"]}% confidence</span>'
                f'<span class="h-date">{h["created_at"][:16].replace("T", " ")}</span>'
                '</div>'
            )
        safe_html(rows)
    else:
        st.caption("No predictions yet. Pick a condition above to get started.")


def main():
    apply_theme()
    sidebar_nav()

    if not st.session_state.logged_in:
        landing_page()
        auth_section()
        return

    home_page()


if __name__ == "__main__":
    main()
