import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flash_attn.attention import FlashAttentionCUDA
import types

def patch_gpt2_attention(model, custom_attn_module):
    """
    Patches the given GPT2 model to use the custom_attn_module.
    The custom module should export `forward(q, k, v, is_causal)`.
    """
    for block in model.transformer.h:
        # We replace the core `_attn` method within GPT2Attention:
        def custom_attn_forward(self, query, key, value, attention_mask=None, head_mask=None):
            # GPT2 `query`, `key`, `value` are of shape (batch, num_heads, seq_len, head_features)
            # Our custom module expects (B, H, N, d) -> exactly matching.
            
            # Since GPT2 is autoregressive, we set is_causal=True
            out = custom_attn_module(query, key, value, is_causal=True)
            
            # GPT2 also expects `attn_weights` in the return structure, but it usually doesn't need them
            # unless `output_attentions` is True. We'll return None for weights.
            return out, None

        # Bind the method to the specific module instance
        block.attn._attn = types.MethodType(custom_attn_forward, block.attn)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    # Load model and set it up for generation
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()
    
    text = "The Hugging Face open source models are"
    inputs = tokenizer(text, return_tensors='pt').to(device)
    
    # 1. Run inference using standard GPT-2 Attention
    print("\n--- STANDARD GPT-2 ATTENTION ---")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=15, do_sample=False)
    print("Output:", tokenizer.decode(out[0], skip_special_tokens=True))
    
    # 2. Patch Attention
    print("\n--- PATCHING WITH CUSTOM FLASH ATTENTION ---")
    # Note: custom module must support forward(q, k, v, is_causal=True)
    try:
        custom_attn = FlashAttentionCUDA(device=device)
        patch_gpt2_attention(model, custom_attn)
        
        # 3. Run inference using Custom Flash Attention
        with torch.no_grad():
            out_custom = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        print("Output:", tokenizer.decode(out_custom[0], skip_special_tokens=True))
        
        # Verify correctness
        if torch.equal(out[0], out_custom[0]):
            print("SUCCESS! Output matches the standard GPT-2 exact output.")
        else:
            print("WARNING: The generation output diverges from standard GPT-2.")
    except Exception as e:
        print(f"Error during custom attention execution: {e}")

if __name__ == '__main__':
    main()
