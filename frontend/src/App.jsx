import { useEffect, useState } from "react";
import { supabase } from "./supabase";
import "./style.css";

const FALLBACK_IMAGE_URL =
  "https://images.unsplash.com/photo-1523906834658-6e24ef2386f9?auto=format&fit=crop&w=1200&q=80";

function normalizePlaceText(value) {
  return (value || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function getBestTitleMatch(searchResults, requestedName) {
  const normalizedRequested = normalizePlaceText(requestedName);

  const scored = (searchResults || [])
    .map((result) => {
      const title = result?.title || "";
      const normalizedTitle = normalizePlaceText(title);

      let score = 0;

      if (normalizedTitle === normalizedRequested) score += 100;
      if (normalizedTitle.includes(normalizedRequested)) score += 40;
      if (normalizedRequested.includes(normalizedTitle)) score += 20;

      const requestedWords = normalizedRequested.split(" ");
      const titleWords = normalizedTitle.split(" ");
      const matchingWords = requestedWords.filter((word) => titleWords.includes(word));
      score += matchingWords.length * 10;

      return { title, score };
    })
    .filter((result) => result.score > 0)
    .sort((a, b) => b.score - a.score);

  return scored[0]?.title || null;
}

async function getPlaceImageUrl(placeName) {
  const name = (placeName || "").trim();

  if (!name) {
    return FALLBACK_IMAGE_URL;
  }

  const searchQuery = `${name} Pune Maharashtra India`;

  try {
    const searchParams = new URLSearchParams({
      action: "query",
      list: "search",
      format: "json",
      formatversion: "2",
      srsearch: searchQuery,
      srlimit: "8",
      origin: "*",
    });

    const searchResponse = await fetch(
      `https://en.wikipedia.org/w/api.php?${searchParams.toString()}`
    );

    if (!searchResponse.ok) {
      return FALLBACK_IMAGE_URL;
    }

    const searchData = await searchResponse.json();
    const searchResults = searchData?.query?.search || [];
    const bestTitle = getBestTitleMatch(searchResults, name);

    if (!bestTitle) {
      return FALLBACK_IMAGE_URL;
    }

    const imageParams = new URLSearchParams({
      action: "query",
      format: "json",
      formatversion: "2",
      prop: "pageimages",
      piprop: "thumbnail",
      pithumbsize: "400",
      titles: bestTitle,
      redirects: "1",
      origin: "*",
    });

    const imageResponse = await fetch(
      `https://en.wikipedia.org/w/api.php?${imageParams.toString()}`
    );

    if (!imageResponse.ok) {
      return FALLBACK_IMAGE_URL;
    }

    const imageData = await imageResponse.json();
    const pages = imageData?.query?.pages || [];
    const page = pages.find((entry) => entry.thumbnail);

    return page?.thumbnail?.source || FALLBACK_IMAGE_URL;
  } catch (error) {
    console.error("Failed to fetch place image:", error);
    return FALLBACK_IMAGE_URL;
  }
}

function App() {
  const [selectedPlace, setSelectedPlace] = useState(null);
  const [showLogin, setShowLogin] = useState(false);
  const [isSignup, setIsSignup] = useState(false);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loginMessage, setLoginMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [user, setUser] = useState(null);
  const [backendPlaces, setBackendPlaces] = useState([]);
  const [crowdResult, setCrowdResult] = useState(null);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoAnalysis, setVideoAnalysis] = useState(null);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploading, setUploading] = useState(false);
  const [placesLoading, setPlacesLoading] = useState(true);
  const [placesError, setPlacesError] = useState("");

  async function getPlacesFromBackend() {
    setPlacesLoading(true);
    setPlacesError("");

    try {
      const response = await fetch("http://127.0.0.1:8000/api/places");

      if (!response.ok) {
        throw new Error("Failed to load tourist places.");
      }

      const data = await response.json();
      console.log("Places from backend:", data);

      const placesWithImages = await Promise.all(
        data.map(async (place) => ({
          ...place,
          image_url: await getPlaceImageUrl(place.name),
        }))
      );

      setBackendPlaces(placesWithImages);
    } catch (error) {
      console.error("Error loading places:", error);
      setPlacesError("Unable to load places right now. Please try again later.");
      setBackendPlaces([]);
    } finally {
      setPlacesLoading(false);
    }
  }
  async function calculateCrowd(peopleCount, capacity) {
  const response = await fetch(
    `http://127.0.0.1:8000/api/crowd/calculate?peopleCount=${peopleCount}&capacity=${capacity}`,
    {
      method: "POST",
    }
  );

  const data = await response.json();

  console.log("Crowd result from backend:", data);

  return data;
}

  async function handleVideoUpload() {
    if (!selectedVideo) {
      setUploadMessage("Please select a video before uploading.");
      return;
    }

    setUploading(true);
    setUploadMessage("Analyzing video...");

    const formData = new FormData();
    formData.append("file", selectedVideo);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/crowd/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (!response.ok || !result.success) {
        throw new Error(result.message || "Upload failed.");
      }

      const analysis = result.analysis || {};
      const peopleCount = Number(analysis.peopleCount ?? 0);

      setVideoAnalysis(analysis);
      setUploadMessage(`${result.message} File: ${result.filename}`);

      if (selectedPlace && selectedPlace.capacity) {
        const crowdData = await calculateCrowd(peopleCount, selectedPlace.capacity);
        setCrowdResult(crowdData);
      }

      setSelectedVideo(null);
      const fileInput = document.getElementById("video-upload-input");
      if (fileInput) {
        fileInput.value = "";
      }
    } catch (error) {
      console.error("Upload error:", error);
      setVideoAnalysis(null);
      setUploadMessage(error.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

 useEffect(() => {
  let isMounted = true;

  async function getSession() {
    const {
      data: { session },
    } = await supabase.auth.getSession();

    if (!isMounted) return;

    setUser(session?.user ?? null);
    setShowLogin(false);
  }

  getSession();

  const {
    data: { subscription },
  } = supabase.auth.onAuthStateChange((_event, session) => {
    const nextUser = session?.user ?? null;
    setUser(nextUser);
    setShowLogin(false);
    setLoginMessage("");
  });

  return () => {
    isMounted = false;
    subscription.unsubscribe();
  };
 }, []);
 useEffect(() => {
  getPlacesFromBackend();
 }, []);

 async function handleLogin(event) {
  event.preventDefault();

  setLoginMessage("");
  setLoading(true);

  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });

  if (error) {
    setLoginMessage(error.message);
    setLoading(false);
    return;
  }

  setEmail("");
  setPassword("");
  setLoginMessage("");
  setShowLogin(false);
  setUser(data.user ?? null);
  setLoading(false);
}

  async function handleSignup(event) {
    event.preventDefault();

    setLoginMessage("");
    setLoading(true);

    if (password !== confirmPassword) {
      setLoginMessage("Passwords do not match.");
      setLoading(false);
      return;
    }

    const { error } = await supabase.auth.signUp({
      email,
      password,
    });

    if (error) {
      setLoginMessage(error.message);
      setLoading(false);
      return;
    }

    setEmail("");
    setPassword("");
    setConfirmPassword("");
    setLoginMessage(
      "Account created! Please check your email to confirm your account."
    );
    setLoading(false);
  }

  async function handleLogout() {
    const { error } = await supabase.auth.signOut();

    if (!error) {
      setUser(null);
      setShowLogin(false);
      setLoginMessage("");
      setEmail("");
      setPassword("");
      setConfirmPassword("");
    }
  }

  function getCrowdLevel(score) {
    if (score <= 30) {
      return "CALM";
    }

    if (score <= 55) {
      return "STEADY";
    }

    if (score <= 78) {
      return "BUSY";
    }

    return "PACKED";
  }

  function calculateCrowdScore(peopleCount, capacity) {
    if (capacity <= 0) {
      return 0;
    }

    const occupancy = (peopleCount / capacity) * 100;

    return Math.min(Math.round(occupancy), 100);
  }

  if (showLogin) {
    return (
      <div className="app">
        <header className="navbar">
          <h2>Pune Crowd Checker</h2>
          <button
            className="login-button"
            onClick={() => {
              setShowLogin(false);
              setIsSignup(false);
              setLoginMessage("");
            }}
          >
            Back
          </button>
        </header>

        <main className="login-page">
          <div className="login-card">
            <p className="small-heading">
              {isSignup ? "WELCOME" : "WELCOME BACK"}
            </p>
            <h1>{isSignup ? "Create Account" : "Login"}</h1>
            <p className="login-description">
              {isSignup
                ? "Create your account to access your Pune Crowd Checker profile."
                : "Login to access your Pune Crowd Checker account."}
            </p>

            <form onSubmit={isSignup ? handleSignup : handleLogin}>
              <label>Email</label>
              <input
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
              />

              <label>Password</label>
              <input
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
              />

              {isSignup && (
                <>
                  <label>Confirm Password</label>
                  <input
                    type="password"
                    placeholder="Confirm your password"
                    value={confirmPassword}
                    onChange={(event) => setConfirmPassword(event.target.value)}
                    required
                  />
                </>
              )}

              <button
                type="submit"
                className="login-submit"
                disabled={loading}
              >
                {loading
                  ? isSignup
                    ? "Creating account..."
                    : "Logging in..."
                  : isSignup
                    ? "Create Account"
                    : "Login"}
              </button>

              {loginMessage && <p className="login-message">{loginMessage}</p>}
            </form>

            {isSignup ? (
              <p className="signup-text">
                Already have an account?{" "}
                <button
                  type="button"
                  className="link-button"
                  onClick={() => {
                    setIsSignup(false);
                    setLoginMessage("");
                    setPassword("");
                    setConfirmPassword("");
                  }}
                >
                  Login
                </button>
              </p>
            ) : (
              <p className="signup-text">
                Don't have an account?{" "}
                <button
                  type="button"
                  className="link-button"
                  onClick={() => {
                    setIsSignup(true);
                    setLoginMessage("");
                    setPassword("");
                    setConfirmPassword("");
                  }}
                >
                  Sign up
                </button>
              </p>
            )}
          </div>
        </main>
      </div>
    );
  }

  if (selectedPlace) {
    const crowdScore = crowdResult ? crowdResult.crowdScore : 0;
    const crowdLevel = crowdResult ? crowdResult.crowdLevel : "CALM";
    const crowdBadgeClass = crowdLevel.toLowerCase();

    return (
      <div className="app">
        <header className="navbar">
          <h2>Pune Crowd Checker</h2>

          <button
            className="login-button"
            onClick={() => setSelectedPlace(null)}
          >
            Back
          </button>
        </header>

        <main className="place-detail-page">
          <div className="place-detail-shell">
            <div className="place-hero">
              <img
                src={selectedPlace.image_url || FALLBACK_IMAGE_URL}
                alt={selectedPlace.name}
                onError={(event) => {
                  event.currentTarget.src = FALLBACK_IMAGE_URL;
                }}
              />
            </div>

            <div className="place-detail-content">
              <div className="place-identity">
                <p className="place-kicker">PUNE TOURIST PLACE</p>
                <h1>{selectedPlace.name}</h1>
                <p className="detail-description">{selectedPlace.description}</p>
              </div>

              <section className="detail-section">
                <div className="section-heading">
                  <h2>About the Place</h2>
                </div>

                <div className="info-grid">
                  <div className="info-card">
                    <span className="info-label">Address</span>
                    <p>{selectedPlace.address}</p>
                  </div>

                  <div className="info-card">
                    <span className="info-label">Opening Hours</span>
                    <p>{selectedPlace.openingHours}</p>
                  </div>

                  <div className="info-card">
                    <span className="info-label">Google Rating</span>
                    <p>
                      {selectedPlace.rating}{" "}
                      {selectedPlace.reviewCount !== "Coming soon" &&
                        `(${selectedPlace.reviewCount} reviews)`}
                    </p>
                  </div>

                  <div className="info-card">
                    <span className="info-label">Entry Fee</span>
                    <p>{selectedPlace.entryFee}</p>
                  </div>
                </div>
              </section>

              <section className="detail-section">
                <div className="section-heading">
                  <h2>Current Crowd</h2>
                </div>

                <div className="crowd-card">
                  <div className="crowd-status-row">
                    <span className={`crowd-level-badge ${crowdBadgeClass}`}>
                      {crowdResult ? crowdResult.crowdLevel : "CALM"}
                    </span>
                  </div>

                  <div className="crowd-score-block">
                    <div className="crowd-score-value">
                      {crowdResult ? `${crowdResult.crowdScore}%` : "0%"}
                    </div>
                    <p className="crowd-score-label">Crowd Score</p>
                  </div>

                  <div className="crowd-meter" aria-label="Crowd score meter">
                    <span style={{ width: `${Math.min(100, Math.max(0, crowdScore))}%` }} />
                  </div>

                  <div className="crowd-summary">
                    <p>People detected: {videoAnalysis ? videoAnalysis.peopleCount : "-"}</p>
                  </div>
                </div>
              </section>

              <section className="detail-section">
                <div className="section-heading">
                  <h2>Analyze this place</h2>
                </div>

                <div className="upload-panel">
                  <p>Upload a crowd video to estimate the current crowd.</p>

                  <div className="upload-control">
                    <input
                      id="video-upload-input"
                      type="file"
                      accept="video/mp4,video/quicktime,video/x-msvideo,video/webm"
                      onChange={(event) => setSelectedVideo(event.target.files[0])}
                    />

                    <button
                      type="button"
                      className="primary-action"
                      onClick={handleVideoUpload}
                      disabled={uploading}
                    >
                      {uploading ? "Analyzing video..." : "Upload Video"}
                    </button>
                  </div>

                  {uploadMessage && <p className="upload-message">{uploadMessage}</p>}
                </div>
              </section>

              <button
                className="secondary-back-button"
                onClick={() => setSelectedPlace(null)}
              >
                ← Back to Places
              </button>
            </div>
          </div>
        </main>
      </div>
    );
  }
async function testBackend() {
  const response = await fetch("http://127.0.0.1:8000/api/test");
  const data = await response.json();

  console.log(data);
 }

  return (
    <div className="app home-page-shell">
      <header className="navbar tourism-navbar">
        <div className="brand-block">
          <h2>CrowdWise Pune</h2>
        </div>

        <nav className="main-navigation" aria-label="Main navigation">
          <a href="#explore">Explore</a>
          <a href="#popular-places">Popular Places</a>
          <a href="#how-it-works">How It Works</a>
        </nav>

        {user ? (
          <div className="user-section">
            <span className="user-email">{user.email}</span>
            <button className="login-button" onClick={handleLogout}>
              Logout
            </button>
          </div>
        ) : (
          <button
            className="login-button"
            onClick={() => {
              setLoginMessage("");
              setShowLogin(true);
            }}
          >
            Login
          </button>
        )}
      </header>

      <main className="homepage-main">
        <section className="hero" id="explore">
          <div className="hero-overlay" />

          <div className="hero-content">
            <p className="small-heading">EXPLORE PUNE SMARTER</p>

            <h1>Know the crowd before you visit.</h1>

            <p className="hero-text">
              Discover Pune's popular places, check estimated crowd levels,
              and plan your visit better.
            </p>

            <div className="hero-search" aria-label="Search places in Pune">
              <input type="text" placeholder="Search places in Pune..." />
              <button type="button">Explore Places</button>
            </div>
          </div>
        </section>

        <section className="value-strip">
          <div className="feature-card">
            <div className="feature-icon">01</div>
            <h3>Check Crowd</h3>
            <p>See the estimated crowd level before visiting.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">02</div>
            <h3>Compare Places</h3>
            <p>Explore popular destinations across Pune.</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">03</div>
            <h3>Plan Better</h3>
            <p>Choose a better time for your visit.</p>
          </div>
        </section>

        <section className="places-section homepage-places" id="popular-places">
          <div className="section-header">
            <div>
              <p className="small-heading section-kicker">DISCOVER</p>
              <h2>Popular Places in Pune</h2>
            </div>
            <p className="section-copy">Explore some of Pune's most popular destinations.</p>
          </div>

          {placesError && (
            <p className="login-message">{placesError}</p>
          )}

          {placesLoading ? (
            <p>Loading places...</p>
          ) : backendPlaces.length === 0 ? (
            <p>No places available right now.</p>
          ) : (
            <div className="places-grid">
              {backendPlaces.map((place) => (
                <div
                  className="place-card"
                  key={place.id}
                  onClick={async () => {
                    setSelectedPlace(place);
                    const result = await calculateCrowd(place.peopleCount, place.capacity);
                    setCrowdResult(result);
                  }}
                >
                  <div className="place-image">
                    <img
                      src={place.image_url || FALLBACK_IMAGE_URL}
                      alt={place.name}
                      loading="lazy"
                      onError={(event) => {
                        event.currentTarget.src = FALLBACK_IMAGE_URL;
                        event.currentTarget.onerror = null;
                      }}
                    />
                  </div>

                  <div className="place-content">
                    <h3>{place.name}</h3>
                    <p>{place.description}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="how-it-works" id="how-it-works">
          <div className="section-header centered-section-header">
            <p className="small-heading section-kicker">HOW IT WORKS</p>
            <h2>How It Works</h2>
          </div>

          <div className="steps-grid">
            <div className="step-card">
              <div className="step-number">01</div>
              <h3>Choose a Place</h3>
              <p>Browse Pune's popular destinations.</p>
            </div>

            <div className="step-card">
              <div className="step-number">02</div>
              <h3>Check the Crowd</h3>
              <p>See the current estimated crowd level.</p>
            </div>

            <div className="step-card">
              <div className="step-number">03</div>
              <h3>Plan Your Visit</h3>
              <p>Choose when and where to visit.</p>
            </div>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="footer-brand">
          <h3>CrowdWise Pune</h3>
          <p>Explore Pune smarter. Plan your visit better.</p>
        </div>

        <div className="footer-links">
          <a href="#explore">Explore</a>
          <a href="#popular-places">Popular Places</a>
          <a href="#how-it-works">How It Works</a>
        </div>

      </footer>
    </div>
  );
}

export default App;
