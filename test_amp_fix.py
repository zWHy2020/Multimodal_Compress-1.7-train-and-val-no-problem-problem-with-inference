"""
测试混合精度训练兼容性修复
"""
import torch
import torch.nn.functional as F

def test_mask_fill_compatibility():
    """测试掩码填充在float16下的兼容性"""
    print("测试掩码填充兼容性...")
    
    # 测试 float16
    scores_fp16 = torch.randn(2, 4, 10, 10, dtype=torch.float16)
    mask = torch.ones(2, 4, 10, 10, dtype=torch.bool)
    mask[0, 0, 0, :5] = 0  # 设置一些掩码位置
    
    # 使用修复后的方法
    fill_value = torch.finfo(scores_fp16.dtype).min if scores_fp16.dtype.is_floating_point else -1e4
    print(f"Fill value for float16: {fill_value}")
    
    try:
        scores_fp16 = scores_fp16.masked_fill(mask == 0, fill_value)
        print("✓ float16 掩码填充成功")
    except Exception as e:
        print(f"✗ float16 掩码填充失败: {e}")
        return False
    
    # 测试 float32
    scores_fp32 = torch.randn(2, 4, 10, 10, dtype=torch.float32)
    fill_value_fp32 = torch.finfo(scores_fp32.dtype).min if scores_fp32.dtype.is_floating_point else -1e4
    print(f"Fill value for float32: {fill_value_fp32}")
    
    try:
        scores_fp32 = scores_fp32.masked_fill(mask == 0, fill_value_fp32)
        print("✓ float32 掩码填充成功")
    except Exception as e:
        print(f"✗ float32 掩码填充失败: {e}")
        return False
    
    # 测试 softmax 后的结果
    try:
        attn_weights = F.softmax(scores_fp16, dim=-1)
        print("✓ float16 softmax 成功")
        print(f"  - 掩码位置的权重: {attn_weights[0, 0, 0, :5].max().item():.6f}")
        print(f"  - 非掩码位置的权重: {attn_weights[0, 0, 0, 5:].max().item():.6f}")
    except Exception as e:
        print(f"✗ float16 softmax 失败: {e}")
        return False
    
    print("\n所有测试通过！")
    return True

if __name__ == '__main__':
    test_mask_fill_compatibility()

