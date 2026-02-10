use std::{fmt::Display, time::Instant};

use burn::{
    config::Config,
    module::{Module, Quantizer},
    nn::{loss::CrossEntropyLossConfig, RotaryEncodingConfig},
    record::{FileRecorder, HalfPrecisionSettings, RecorderError},
    tensor::{
        backend::{AutodiffBackend, Backend},
        quantization::{Calibration, QuantScheme},
        Device, Int, Shape, Tensor, TensorData,
    },
    train::{InferenceStep, ItemLazy, TrainOutput, TrainStep},
};

use crate::{
    nn::{
        pos_encoding::{PositionalEncodingState, RopeConfig, RopeFrequencyScaling},
        transformer::{Transformer, TransformerCache, TransformerConfig},
    },
    tokenizer::Tokenizer,
};

#[cfg(feature = "tiny")]
use crate::tokenizer::SentencePieceTokenizer;
#[cfg(feature = "llama3")]
use crate::tokenizer::Tiktoken;

#[derive(Clone, Debug, Default)]
/// Llama-3 model variants to load.
pub enum LlamaVersion {
    /// Llama-3-8B-Instruct.
    Llama3Instruct,
    /// Llama-3.1-8B-Instruct.
    Llama31Instruct,
    /// Llama-3.2-3B-Instruct.
    Llama323bInstruct,
    #[default]
    /// Llama-3.2-1B-Instruct.
    Llama321bInstruct,
    /// Llama-3.2-1B-Instruct 4-bit quantized with block size 32.
    Llama321bInstructQ4FB32,
}

/// Tiny Llama model variants to load.
pub enum TinyLlamaVersion {
    /// TinyLlama-1.1B-Chat-v1.0
    V1,
}

#[derive(Config, Debug)]
pub struct LlamaConfig {
    /// The size of the model.
    #[config(default = "4096")]
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of transformer blocks.
    #[config(default = "32")]
    pub num_hidden_layers: usize,
    /// The number of attention heads.
    #[config(default = "32")]
    pub num_attention_heads: usize,
    /// The number of key-value heads.
    pub num_key_value_heads: Option<usize>,
    /// The vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon
    #[config(default = "1e-5")]
    pub norm_eps: f64,
    /// Rotary positional encoding (RoPE).
    #[config(default = "RopeConfig::new(10000.0)")]
    pub rope: RopeConfig,
    /// Maximum sequence length for input text.
    #[config(default = "128")]
    pub max_seq_len: usize,
    /// Maximum batch size (used for key-value cache).
    #[config(default = "1")]
    pub max_batch_size: usize,
    /// The tokenizer path.
    pub tokenizer: String,
}

impl LlamaConfig {
    /// Llama-3.2-3B configuration.
    pub fn llama3_2_3b(tokenizer_path: &str) -> Self {
        // hidden_size = 8192; vocab_size = 128256
        Self::new(8192, 128256, tokenizer_path.to_string())
            .with_d_model(3072)
            .with_num_hidden_layers(28)
            .with_num_attention_heads(24)
            .with_num_key_value_heads(Some(8))
            .with_rope(
                RopeConfig::new(500000.0)
                    .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.))),
            )
    }

    /// Llama-3.2-3B configuration for testing..
    pub fn llama3_2_1b_test() -> Self {
        // hidden_size = 8192; vocab_size = bytes == 255
        Self::new(128, 255, "test".to_string())
            .with_d_model(64)
            .with_num_hidden_layers(2)
            .with_num_attention_heads(4)
            .with_num_key_value_heads(Some(2))
            .with_rope(
                RopeConfig::new(500000.0)
                    .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.))),
            )
    }

    /// Llama-3.2-1B configuration.
    pub fn llama3_2_1b(tokenizer_path: &str) -> Self {
        // hidden_size = 8192; vocab_size = 128256
        Self::new(8192, 128256, tokenizer_path.to_string())
            .with_d_model(2048)
            .with_num_hidden_layers(16)
            .with_num_key_value_heads(Some(8))
            .with_rope(
                RopeConfig::new(500000.0)
                    .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.))),
            )
    }

    /// Llama-3.1-8B configuration.
    pub fn llama3_1_8b(tokenizer_path: &str) -> Self {
        // hidden_size = 14336; vocab_size = 128256
        Self::new(14336, 128256, tokenizer_path.to_string())
            .with_num_key_value_heads(Some(8))
            .with_rope(RopeConfig::new(500000.0).with_scaled(Some(RopeFrequencyScaling::new())))
    }

    /// Llama-3-8B configuration.
    pub fn llama3_8b(tokenizer_path: &str) -> Self {
        // hidden_size = 14336; vocab_size = 128256
        Self::new(14336, 128256, tokenizer_path.to_string())
            .with_num_key_value_heads(Some(8))
            .with_rope(RopeConfig::new(500000.0))
    }

    /// TinyLlama-1.1B Chat v1.0 configuration.
    pub fn tiny_llama(tokenizer_path: &str) -> Self {
        // hidden_size = 5632; vocab_size = 32000
        Self::new(5632, 32000, tokenizer_path.to_string())
            .with_d_model(2048)
            .with_num_hidden_layers(22)
            .with_num_key_value_heads(Some(4))
            .with_rope(RopeConfig::new(10000.0))
    }

    /// Load pre-trained Llama-3.2-3B model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    #[cfg(feature = "llama3")]
    pub fn load_llama3_2_3b<B: Backend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        use burn::record::NamedMpkFileRecorder;

        let llama = Self::llama3_2_3b(tokenizer_path)
            .with_max_seq_len(max_seq_len)
            .init::<B, Tiktoken>(device)?;

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

        Ok(llama)
    }

    /// Load pre-trained Llama-3.2-1B model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    #[cfg(feature = "llama3")]
    pub fn load_llama3_2_1b<B: Backend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        use burn::record::NamedMpkFileRecorder;

        let llama = Self::llama3_2_1b(tokenizer_path)
            .with_max_seq_len(max_seq_len)
            .init::<B, Tiktoken>(device)?;

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

        Ok(llama)
    }

    /// Load pre-trained Llama-3.1-8B model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    #[cfg(feature = "llama3")]
    pub fn load_llama3_1_8b<B: Backend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        use burn::record::NamedMpkFileRecorder;

        let llama = Self::llama3_1_8b(tokenizer_path)
            .with_max_seq_len(max_seq_len)
            .init::<B, Tiktoken>(device)?;

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

        Ok(llama)
    }

    /// Load pre-trained Llama-3-8B model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    #[cfg(feature = "llama3")]
    pub fn load_llama3_8b<B: Backend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        use burn::record::NamedMpkFileRecorder;

        let llama = Self::llama3_8b(tokenizer_path)
            .with_max_seq_len(max_seq_len)
            .init::<B, Tiktoken>(device)?;

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

        Ok(llama)
    }

    /// Load pre-trained TinyLlama-1.1B Chat v1.0 model with [SentenciePiece](https://github.com/google/sentencepiece) tokenizer.
    #[cfg(feature = "tiny")]
    pub fn load_tiny_llama<B: Backend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, SentencePieceTokenizer>, String> {
        use burn::record::NamedMpkFileRecorder;

        let llama = Self::tiny_llama(tokenizer_path)
            .with_max_seq_len(max_seq_len)
            .init::<B, SentencePieceTokenizer>(device)?;

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

        Ok(llama)
    }

    /// Initialize a new [Llama](Llama) module.
    pub fn init<B: Backend, T: Tokenizer>(
        &self,
        device: &Device<B>,
    ) -> Result<Llama<B, T>, String> {
        let tokenizer = T::new(&self.tokenizer)?;
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        let config = TransformerConfig::new(
            self.vocab_size,
            self.num_hidden_layers,
            self.d_model,
            self.hidden_size,
            self.num_attention_heads,
            num_key_value_heads,
        )
        .with_max_seq_len(self.max_seq_len)
        .with_norm_eps(self.norm_eps);

        let model = config.init(device);
        let cache = TransformerCache::new(&config, self.max_batch_size, device);

        let rope = RotaryEncodingConfig::new(
            self.max_seq_len * 5,
            self.d_model / self.num_attention_heads,
        )
        .with_theta(self.rope.theta);

        let rope = if let Some(scaling) = &self.rope.scaled {
            let freq_scaling_fn = move |x| scaling.freq_scaling_by_parts(x);
            rope.init_with_frequency_scaling(freq_scaling_fn, device)
        } else {
            rope.init(device)
        };

        let pos_encoding = PositionalEncodingState::new(rope);

        Ok(Llama {
            tokenizer,
            model,
            cache,
            pos_encoding,
            device: device.clone(),
        })
    }
}

/// Meta Llama large language model and tokenizer.
#[derive(Debug)]
pub struct Llama<B: Backend, T: Tokenizer> {
    /// The tokenizer.
    pub tokenizer: T,
    /// Llama decoder-only transformer.
    pub model: Transformer<B>,
    /// Key-value cache for each transformer block.
    pub cache: TransformerCache<B>,
    /// Rotary positional encoding (RoPE).
    pub pos_encoding: PositionalEncodingState<B>,
    pub device: Device<B>,
}

impl<B: Backend, T: Tokenizer> Llama<B, T> {
    /// Encode a string into a tensor of tokens.
    pub fn tokenize(&self, text: &str) -> Tensor<B, 1, Int> {
        let tokens = self.tokenizer.encode(text, false, false);

        let shape = Shape::new([tokens.len()]);
        Tensor::<B, 1, Int>::from_data(TensorData::new(tokens, shape), &self.device)
    }

    /// Save Llama model to file using the specified recorder.
    pub fn save<R: FileRecorder<B>>(
        self,
        file_path: &str,
        recorder: &R,
    ) -> Result<(), RecorderError> {
        println!("Saving record...");
        let now = Instant::now();
        self.model.save_file(file_path, recorder)?;
        let elapsed = now.elapsed().as_secs();
        println!("Saved in {elapsed}s");

        Ok(())
    }

    /// Load Llama model from file using the specified recorder.
    pub fn load<R: FileRecorder<B>>(
        mut self,
        file_path: &str,
        recorder: &R,
    ) -> Result<Self, RecorderError> {
        self.model = self.model.load_file(file_path, recorder, &self.device)?;
        Ok(self)
    }

    /// Reset the model state (used between generations)
    pub fn reset(&mut self) {
        self.cache.reset()
    }

    /// Quantize the model weights.
    pub fn quantize(mut self, scheme: QuantScheme) -> Self {
        let calibration = Calibration::MinMax;
        let mut quantizer = Quantizer {
            calibration,
            scheme,
        };
        let device = &self.model.devices()[0];

        // TODO: improve module mapper usage for quantization (currently, this leads to additional memory usage)
        // self.model = self.model.quantize_weights(&mut quantizer);

        // Quantizing by layer reduces the peak memory usage
        let mut layers = Vec::with_capacity(self.model.layers.len());
        for layer in self.model.layers.drain(..) {
            layers.push(layer.quantize_weights(&mut quantizer));
        }
        self.model.layers = layers;
        B::memory_cleanup(device);

        self.model.tok_embeddings = self.model.tok_embeddings.quantize_weights(&mut quantizer);
        self.model.output = self.model.output.quantize_weights(&mut quantizer);

        self
    }
}

// impl<B: Backend, T: Tokenizer> Display for Llama<B, T> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", "llama 3/tiny")
//     }
// }

// impl<B: Backend, T: Tokenizer> InferenceStep for Llama<B, T> {
//     type Input = LlamaInput<B>;
//     type Output = LlamaOutput<B>;

//     fn step(&self, item: LlamaInput<B>) -> LlamaOutput<B> {
//         let logits = self.model.forward_train(item.tokens, &self.pos_encoding);
//         let [batch_size, seq_len, vocab_size] = logits.dims();
//         let logits_flattened = logits.clone().reshape([batch_size * seq_len, vocab_size]);
//         let targets_flattened = item.targets.reshape([batch_size * seq_len]);
//         let loss = CrossEntropyLossConfig::new()
//             .init(&logits.device())
//             .forward(logits_flattened, targets_flattened);

//         LlamaOutput { loss, logits }
//     }
// }

// impl<B: AutodiffBackend, T: Tokenizer> TrainStep for Llama<B, T> {
//     type Input = LlamaInput<B>;
//     type Output = LlamaOutput<B>;

//     fn step(&self, item: LlamaInput<B>) -> TrainOutput<LlamaOutput<B>> {
//         let logits = self.model.forward_train(item.tokens, &self.pos_encoding);

//         let [batch_size, seq_len, vocab_size] = logits.dims();
//         let logits_flattened = logits.clone().reshape([batch_size * seq_len, vocab_size]);
//         let targets_flattened = item.targets.reshape([batch_size * seq_len]);

//         let loss = CrossEntropyLossConfig::new()
//             .init(&logits.device())
//             .forward(logits_flattened, targets_flattened);
//         let grads = loss.backward();

//         let output = LlamaOutput { loss, logits };
//         TrainOutput::new(&self.model, grads, output)
//     }
// }

// #[derive(Debug, Clone)]
// pub struct LlamaInput<B: Backend> {
//     /// [batch_size, seq_len]
//     pub tokens: Tensor<B, 2, Int>,
//     /// [batch_size, seq_len]
//     pub targets: Tensor<B, 2, Int>,
// }

// #[derive(Debug, Clone)]
// pub struct LlamaOutput<B: Backend> {
//     pub loss: Tensor<B, 1>,
//     /// [batch_size, seq_len, vocab_size]
//     pub logits: Tensor<B, 3>,
// }

// impl<B: Backend> ItemLazy for LlamaOutput<B> {
//     type ItemSync = Self;

//     fn sync(self) -> Self::ItemSync {
//         self
//     }
// }
