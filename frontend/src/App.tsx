import React, { useState } from 'react';
import axios from 'axios';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Upload, Cpu, UserCheck, BarChart3, Loader2, ArrowRight, Maximize2, Minimize2 } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const App = () => {
  const [activeTab, setActiveTab] = useState<'pca' | 'esm' | 'docs'>('pca');

  return (
    <div className="min-h-screen bg-white text-[#1a1a1a] font-sans selection:bg-[#1a1a1a] selection:text-white">
      {/* Editorial Header */}
      <header className="px-6 md:px-12 py-12 border-b border-[#eeeeee]">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-end gap-8">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <span className="text-[10px] font-bold tracking-[0.2em] uppercase border border-[#1a1a1a] px-2 py-0.5">Lab 01</span>
              <h1 className="text-4xl font-light tracking-tight italic">PCA Vision</h1>
            </div>
            <p className="text-sm text-[#777] max-w-md font-medium leading-relaxed">
              Computational linear algebra for image decomposition and facial subspace recognition. 
              Precision-focused principal component analysis.
            </p>
          </div>
          
          <nav className="flex gap-1 bg-[#f5f5f5] p-1 rounded-sm">
            <button 
              onClick={() => setActiveTab('pca')}
              className={`px-6 py-2 text-[11px] font-bold uppercase tracking-widest transition-all ${activeTab === 'pca' ? 'bg-white shadow-sm text-[#1a1a1a]' : 'text-[#999] hover:text-[#1a1a1a]'}`}
            >
              Compression
            </button>
            <button 
              onClick={() => setActiveTab('esm')}
              className={`px-6 py-2 text-[11px] font-bold uppercase tracking-widest transition-all ${activeTab === 'esm' ? 'bg-white shadow-sm text-[#1a1a1a]' : 'text-[#999] hover:text-[#1a1a1a]'}`}
            >
              Recognition
            </button>
            <button 
              onClick={() => setActiveTab('docs')}
              className={`px-6 py-2 text-[11px] font-bold uppercase tracking-widest transition-all ${activeTab === 'docs' ? 'bg-white shadow-sm text-[#1a1a1a]' : 'text-[#999] hover:text-[#1a1a1a]'}`}
            >
              Docs
            </button>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 md:px-12 py-12">
        {activeTab === 'pca' ? <PCATab /> : activeTab === 'esm' ? <ESMTab /> : <DocsTab />}
      </main>

      <footer className="px-12 py-12 border-t border-[#eeeeee] flex justify-between items-center mt-24">
        <span className="text-[10px] font-bold uppercase tracking-[0.3em] text-[#bbb]">Linear Algebra Matrix</span>
        <span className="text-[10px] font-bold uppercase tracking-[0.3em] text-[#bbb]">© 2026 Studio PCA</span>
      </footer>
    </div>
  );
};

const PCATab = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [compressed, setCompressed] = useState<string | null>(null);
  const [k, setK] = useState(50);
  const [loading, setLoading] = useState(false);
  const [varianceData, setVarianceData] = useState<{name: number, value: number}[]>([]);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const f = e.target.files[0];
      setFile(f);
      setPreview(URL.createObjectURL(f));
    }
  };

  const runPCA = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('k', k.toString());
    try {
      const { data } = await axios.post(`${API_BASE}/api/compress`, formData);
      setCompressed(data.compressed_image);
      setVarianceData(data.variance_data.map((v: number, i: number) => ({ name: i + 1, value: v * 100 })));
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-16 animate-in fade-in slide-in-from-bottom-2 duration-700">
      {/* Controls */}
      <div className="lg:col-span-4 space-y-12">
        <section className="space-y-4">
          <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[#999]">Source Image</label>
          <div className="relative aspect-square border border-[#eeeeee] flex items-center justify-center group overflow-hidden bg-[#fafafa]">
            {preview ? (
              <img src={preview} className="w-full h-full object-cover grayscale transition-all duration-500 group-hover:grayscale-0 group-hover:scale-105" />
            ) : (
              <div className="text-center space-y-2">
                <Upload className="w-5 h-5 mx-auto text-[#ccc]" />
                <p className="text-[10px] font-bold uppercase tracking-widest text-[#bbb]">Click to Import</p>
              </div>
            )}
            <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" onChange={handleUpload} />
          </div>
        </section>

        <section className="space-y-6">
          <div className="flex justify-between items-end">
            <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[#999]">Eigenvalue $k$</label>
            <span className="text-4xl font-light italic">{k}</span>
          </div>
          <input 
            type="range" min="1" max="200" value={k} 
            onChange={(e) => setK(parseInt(e.target.value))}
            className="w-full h-[1px] bg-[#ddd] appearance-none cursor-pointer accent-[#1a1a1a]"
          />
          <button 
            onClick={runPCA}
            disabled={loading || !file}
            className="w-full py-4 border border-[#1a1a1a] text-[11px] font-bold uppercase tracking-[0.3em] hover:bg-[#1a1a1a] hover:text-white transition-all disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-[#1a1a1a] flex items-center justify-center gap-3"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <>Execute Decomposition <ArrowRight className="w-3 h-3" /></>}
          </button>
        </section>
      </div>

      {/* Results */}
      <div className="lg:col-span-8 space-y-16">
        <section className="grid grid-cols-1 md:grid-cols-2 gap-px bg-[#eeeeee] border border-[#eeeeee]">
          <div className="bg-white p-8 space-y-6">
            <div className="flex justify-between items-center">
              <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-[#999]">Original Matrix</span>
              <Minimize2 className="w-3 h-3 text-[#ddd]" />
            </div>
            <div className="aspect-video bg-[#fafafa] overflow-hidden flex items-center justify-center">
              {preview ? <img src={preview} className="w-full h-full object-cover grayscale hover:grayscale-0 transition-all duration-700" /> : <div className="text-[#eee] font-light text-6xl">01</div>}
            </div>
          </div>
          <div className="bg-white p-8 space-y-6">
            <div className="flex justify-between items-center">
              <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-[#1a1a1a]">Reconstructed</span>
              <Maximize2 className="w-3 h-3 text-[#1a1a1a]" />
            </div>
            <div className="aspect-video bg-[#fafafa] overflow-hidden flex items-center justify-center">
              {compressed ? <img src={compressed} className="w-full h-full object-cover" /> : <div className="text-[#eee] font-light text-6xl">02</div>}
            </div>
          </div>
        </section>

        <section className="space-y-8 border-t border-[#eeeeee] pt-12">
          <div className="flex items-center gap-4">
            <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[#999]">Exploratory Data Analysis</label>
            <div className="h-[1px] flex-1 bg-[#eeeeee]" />
          </div>
          <div className="h-64 w-full">
            {varianceData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={varianceData} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                  <XAxis dataKey="name" hide />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#bbb' }} unit="%" />
                  <Tooltip 
                    contentStyle={{ border: '1px solid #eee', borderRadius: '0', boxShadow: 'none', fontSize: '10px', fontWeight: 'bold', textTransform: 'uppercase' }}
                  />
                  <Area type="stepAfter" dataKey="value" stroke="#1a1a1a" fill="#1a1a1a" fillOpacity={0.03} strokeWidth={1} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-[10px] font-bold uppercase tracking-[0.2em] text-[#ddd]">
                Await execution for statistical projection
              </div>
            )}
          </div>
          <p className="text-[11px] text-[#777] leading-relaxed max-w-xl italic">
            Cumulative Explained Variance ratio across the spectrum. A higher $k$ integer increases fidelity by encompassing more variance within the subspace.
          </p>
        </section>
      </div>
    </div>
  );
};

const ESMTab = () => {
  const [child, setChild] = useState<{f: File, p: string} | null>(null);
  const [adult, setAdult] = useState<{f: File, p: string} | null>(null);
  const [result, setResult] = useState<{similarity: number, distance: number} | null>(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);

  const train = async () => {
    setTraining(true);
    try {
      await axios.post(`${API_BASE}/api/esm/train`);
      setTrained(true);
    } catch (e) { console.error(e); } finally { setTraining(false); }
  };

  const compare = async () => {
    if (!child || !adult) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('child_file', child.f);
    formData.append('adult_file', adult.f);
    try {
      const { data } = await axios.post(`${API_BASE}/api/esm/compare`, formData);
      setResult(data);
    } catch (e) { console.error(e); } finally { setLoading(false); }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-24 animate-in fade-in slide-in-from-bottom-2 duration-700">
      <section className="flex flex-col md:flex-row items-center justify-between border-b border-[#eeeeee] pb-12 gap-8">
        <div className="space-y-4">
          <h2 className="text-3xl font-light italic">Eigenface Subspace</h2>
          <p className="text-[11px] font-bold uppercase tracking-widest text-[#999]">Database: Olivetti Human Faces</p>
        </div>
        <button 
          onClick={train} disabled={training || trained}
          className="px-10 py-4 border border-[#1a1a1a] text-[11px] font-bold uppercase tracking-[0.3em] hover:bg-[#1a1a1a] hover:text-white transition-all disabled:opacity-20 flex items-center gap-3"
        >
          {training ? <Loader2 className="w-4 h-4 animate-spin" /> : trained ? 'Ready' : 'Initialize Subspace'}
        </button>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
        <FaceBox label="Matrix Alpha (Temporal T0)" val={child} set={setChild} />
        <FaceBox label="Matrix Beta (Temporal T1)" val={adult} set={setAdult} />
      </div>

      <div className="flex flex-col items-center gap-12">
        <button 
          onClick={compare}
          disabled={!child || !adult || loading || !trained}
          className="px-16 py-6 border border-[#1a1a1a] text-[12px] font-bold uppercase tracking-[0.4em] hover:bg-[#1a1a1a] hover:text-white transition-all disabled:opacity-20 flex items-center gap-4"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <>Compute Subspace Distance <UserCheck className="w-4 h-4" /></>}
        </button>

        {result && (
          <div className="w-full space-y-8 animate-in zoom-in-95 duration-500 text-center">
            <div className="space-y-2">
              <span className="text-[10px] font-bold uppercase tracking-[0.4em] text-[#bbb]">Similarity Coefficient</span>
              <div className="text-8xl font-light tracking-tighter italic">{result.similarity}%</div>
            </div>
            <div className="max-w-md mx-auto h-[1px] bg-[#eeeeee] relative">
              <div className="absolute top-1/2 left-0 h-[3px] bg-[#1a1a1a] -translate-y-1/2 transition-all duration-1000" style={{ width: `${result.similarity}%` }} />
            </div>
            <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-[#bbb]">Euclidean Delta: {result.distance}</p>
          </div>
        )}
      </div>
    </div>
  );
};

const FaceBox = ({ label, val, set }: any) => (
  <div className="space-y-4">
    <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[#999]">{label}</label>
    <div className="aspect-[4/5] bg-[#fafafa] border border-[#eeeeee] relative overflow-hidden group">
      {val ? (
        <img src={val.p} className="w-full h-full object-cover grayscale transition-all duration-500 group-hover:grayscale-0" />
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-[#ddd] space-y-4">
          <Upload className="w-6 h-6" />
          <span className="text-[10px] font-bold uppercase tracking-widest">Import Matrix</span>
        </div>
      )}
      <input 
        type="file" className="absolute inset-0 opacity-0 cursor-pointer" 
        onChange={(e) => e.target.files?.[0] && set({ f: e.target.files[0], p: URL.createObjectURL(e.target.files[0]) })} 
      />
    </div>
  </div>
);

const DocsTab = () => (
  <div className="max-w-3xl mx-auto space-y-24 animate-in fade-in slide-in-from-bottom-2 duration-700">
    <section className="space-y-8">
      <h2 className="text-4xl font-light italic tracking-tight">System Documentation</h2>
      <div className="h-[1px] w-24 bg-[#1a1a1a]" />
      <p className="text-[#777] leading-relaxed italic">
        This platform utilizes Principal Component Analysis (PCA) to decompose high-dimensional visual data into significant mathematical structures.
      </p>
    </section>

    <div className="grid grid-cols-1 gap-24">
      <article className="space-y-6">
        <span className="text-[10px] font-bold uppercase tracking-[0.4em] text-[#bbb]">Module 01</span>
        <h3 className="text-2xl font-light italic">Image Compression (PCA)</h3>
        <div className="space-y-4 text-sm text-[#555] leading-relaxed">
          <p>
            Image compression via PCA works by identifying the "Principal Components" (eigenvectors) of the image matrix. 
            By retaining only the top $k$ components with the highest variance (eigenvalues), we can reconstruct the image using significantly less data.
          </p>
          <ul className="space-y-2 list-none border-l border-[#eee] pl-6 py-2">
            <li><strong className="text-[#1a1a1a] uppercase text-[10px] tracking-widest block">Input $k$</strong> Lower values yield high compression but lossy results. Higher values preserve fidelity.</li>
            <li><strong className="text-[#1a1a1a] uppercase text-[10px] tracking-widest block">EDA Chart</strong> Displays the cumulative explained variance. Aim for the "elbow" where adding more components provides diminishing returns.</li>
          </ul>
        </div>
      </article>

      <article className="space-y-6 text-right">
        <span className="text-[10px] font-bold uppercase tracking-[0.4em] text-[#bbb]">Module 02</span>
        <h3 className="text-2xl font-light italic text-right">Face Recognition (ESM)</h3>
        <div className="space-y-4 text-sm text-[#555] leading-relaxed">
          <p>
            The Eigenface Subspace Method (ESM) projects new face images into a mathematical "face space" constructed from a training dataset (Olivetti Faces).
          </p>
          <p>
            Comparison is achieved by measuring the Euclidean distance between the projections of two different temporal states (e.g., childhood vs. adult) within this subspace.
          </p>
          <div className="inline-block border-r border-[#eee] pr-6 py-2 text-right">
            <strong className="text-[#1a1a1a] uppercase text-[10px] tracking-widest block">Subspace Training</strong> Must be initialized before comparison to compute the mean face and basis vectors.
          </div>
        </div>
      </article>

      <article className="space-y-6 pt-12 border-t border-[#eee]">
        <h3 className="text-lg font-bold uppercase tracking-[0.2em] text-[#1a1a1a]">Usage Protocol</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-[11px] text-[#888] font-medium tracking-wide leading-relaxed">
          <div className="space-y-2">
            <span className="text-[#1a1a1a] font-bold italic">01. Import</span>
            <p>Upload source matrices (JPG/PNG) into the relevant laboratory module.</p>
          </div>
          <div className="space-y-2">
            <span className="text-[#1a1a1a] font-bold italic">02. Initialize</span>
            <p>Set parameters ($k$ value) or train the eigenspace basis for face recognition.</p>
          </div>
          <div className="space-y-2">
            <span className="text-[#1a1a1a] font-bold italic">03. Analyze</span>
            <p>Inspect statistical charts and reconstruction deltas for precision validation.</p>
          </div>
        </div>
      </article>
    </div>
  </div>
);

export default App;
