{ config, pkgs, lib, ... }:

{
  # CUDA toolkit for GPU compute
  environment.systemPackages = with pkgs; [
    # CUDA development
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    cudaPackages.tensorrt
    
    # LLM runners with CPU/GPU hybrid support
    ollama  # Supports CPU and GPU/CPU split
    llama-cpp  # Efficient CPU/GPU inference
    
    # Python ML packages (base environment)
    (python312.withPackages (ps: with ps; [
      # Jupyter for experimentation
      jupyter
      ipykernel
      notebook
      
      # Basic ML packages
      numpy
      pandas
      matplotlib
      scikit-learn
      
      # Deep learning frameworks
      torch-bin  # PyTorch with CUDA
      torchvision-bin
      tensorflow  # If you need both
      
      # LLM specific
      transformers
      accelerate
      datasets
      tokenizers
      sentencepiece
      
      # Fine-tuning tools
      peft  # Parameter efficient fine-tuning
      bitsandbytes  # Quantization
      
      # Quantization for RAM inference
      auto-gptq  # GPTQ quantization
      optimum  # Optimization library
      
      # CPU inference optimization
      onnxruntime  # CPU optimized inference
      ctranslate2  # Fast CPU inference
      
      # Utilities
      tqdm
      wandb  # Experiment tracking
      tensorboard
      psutil  # Monitor RAM usage
      
      # API/Serving
      fastapi
      uvicorn
      gradio  # Quick web UIs
    ]))
    
    # C++ libraries for efficient CPU inference
    mkl  # Intel Math Kernel Library (great for Intel CPUs)
    
    # Model conversion tools
    protobuf  # For model formats
    onnx  # Model interchange
    
    # Monitoring
    nvitop  # Better nvidia-smi for ML
    gpustat  # Simple GPU monitoring
    
    # Data processing
    jq  # JSON processing
    yq  # YAML processing
    miller  # CSV/JSON/etc processing
  ];
  
  # Jupyter service (optional - runs notebook server)
  services.jupyter = {
    enable = true;
    password = "'sha1:...'";  # Generate with: jupyter notebook password
    port = 8888;
    user = "yourusername";  # Change to your username
    group = "users";
    notebookConfig = ''
      c.NotebookApp.open_browser = False
      c.NotebookApp.ip = '0.0.0.0'
    '';
  };
  
  # Ollama service with CPU/GPU configuration
  systemd.services.ollama = {
    description = "Ollama LLM Server";
    after = [ "network.target" ];
    wantedBy = [ "multi-user.target" ];
    
    serviceConfig = {
      Type = "simple";
      ExecStart = "${pkgs.ollama}/bin/ollama serve";
      Restart = "always";
      User = "yourusername";  # Change to your username
      
      # Memory limits for safety
      MemoryMax = "110G";  # Leave some RAM for system
      
      # Pin to P-cores for better performance (cores 0-11)
      CPUAffinity = "0-11";
      
      Environment = [
        "OLLAMA_HOST=0.0.0.0:11434"
        "CUDA_VISIBLE_DEVICES=0"
        "OLLAMA_MODELS=/var/lib/models/ollama"
        "OLLAMA_NUM_GPU=999"  # Use all GPU layers possible
        "OLLAMA_CPU_THREADS=12"  # Use P-cores only (12 threads)
      ];
    };
  };
  
  # Create model directories
  system.activationScripts.llm-dirs = ''
    mkdir -p /var/lib/models/ollama
    mkdir -p /var/lib/models/huggingface
    mkdir -p /var/lib/models/gguf  # For llama.cpp models
    mkdir -p /var/lib/models/checkpoints
    chown -R yourusername:users /var/lib/models  # Change username
  '';
  
  # Environment variables
  environment.variables = {
    TRANSFORMERS_CACHE = "/var/lib/models/huggingface";
    HF_HOME = "/var/lib/models/huggingface";
    CUDA_VISIBLE_DEVICES = "0";
    
    # CPU optimization for 13600KF
    OMP_NUM_THREADS = "12";  # P-cores only for compute
    MKL_NUM_THREADS = "12";
    OPENBLAS_NUM_THREADS = "12";
    
    # Allow transformers to use all available RAM
    PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True";
  };
  
  # Kernel parameters for large models
  boot.kernel.sysctl = {
    # Allow memory overcommit for large models
    "vm.overcommit_memory" = 1;
    # Increase for model memory mapping
    "vm.max_map_count" = 2147483642;
    # Disable transparent huge pages for better control
    "kernel.transparent_hugepage" = "never";
  };
  
  # CPU governor for consistent performance
  powerManagement.cpuFreqGovernor = "performance";
}
