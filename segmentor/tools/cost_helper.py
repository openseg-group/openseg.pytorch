import math


def cost_conv(H, W, Cin, Cout, kernel_size, bias=True, stride=1, padding=0, dilation=1):
    flatten_ksize = (kernel_size - 1) * dilation + 1
    oh = (H + padding - flatten_ksize // 2) // stride
    ow = (W + padding - flatten_ksize // 2) // stride
    bias_cost = 1 if bias else 0
    cost = oh * ow * (Cin * kernel_size ** 2 * 2 + bias_cost) * Cout
    return cost


def cost_affinity(N, C, num_subsample=None):
    if num_subsample is None:
        num_subsample = N
    return N * num_subsample * C * 2


def cost_ia(H, W, CI, CK, CV, CO, P=8):
    dh = dw = P
    cw = cost_conv(H, W, CV, CO, 1)
    cqkv = cost_conv(H, W, CI, CK, 1) * 2 + cost_conv(H, W, CI, CV, 1)
    cw *= 2  # long and short
    cqkv *= 2
    oh, ow = math.ceil(H / dh), math.ceil(W / dw)
    caffinity = (
        cost_affinity(oh * ow, CK) * dh * dw + cost_affinity(dh * dw, CK) * oh * ow
    )
    caggregate = (
        cost_affinity(oh * ow, CV) * dh * dw + cost_affinity(dh * dw, CV) * oh * ow
    )
    c_skip = cost_conv(H, W, CI + CO, CO, 1)
    c_ia = cw + cqkv + caffinity + caggregate + c_skip
    return c_ia


def cost_sa(H, W, CI, CK, CV, CO):
    cqkv = cost_conv(H, W, CI, CK, 1) * 2 + cost_conv(H, W, CI, CV, 1)
    cw = cost_conv(H, W, CV, CO, 1)
    caffinity = cost_affinity(H * W, CK)
    caggregate = cost_affinity(H * W, CV)
    c_skip = cost_conv(H, W, CI + CO, CO, 1)
    c_nl = cw + cqkv + caffinity + caggregate + c_skip
    return c_nl


def cost_rcca(H, W, CI, CInter, CK, CV, CO):
    c_conva = cost_conv(H, W, 2048, CInter, 3, padding=1, bias=False)

    cqkv = cost_conv(H, W, CInter, CK, 1) * 2 + cost_conv(H, W, CInter, CV, 1)
    caffinity = cost_affinity(H * W, CK, num_subsample=(H + W - 1))
    caggregate = cost_affinity(H * W, CV, num_subsample=(H + W - 1))

    c_convb = cost_conv(H, W, CO, CO, 3, padding=1, bias=False)

    c_bottleneck = cost_conv(H, W, CI + CO, CO, 3, padding=1, bias=False)

    c_cc = (cqkv + caffinity + caggregate) * 2 + c_conva + c_convb + c_bottleneck

    return c_cc


def cost_double_attention(H, W, CI, factor, global_cnt):
    c_gather_dist = (
        cost_conv(H, W, CI, global_cnt, 1) + cost_conv(H, W, global_cnt, global_cnt, 1)
    ) * 2
    c_down_up = cost_conv(H, W, CI, CI // factor, 1) * 2

    c_mul_gather = global_cnt * H * W * (CI // factor)
    c_mul_dist = global_cnt * H * W * (CI // factor)

    c_skip = cost_conv(H, W, CI + CI, CI, 1)
    c_da = c_gather_dist + c_down_up + c_mul_gather + c_mul_dist + c_skip
    return c_da


def cost_ocr(H, W, CI, CK, CV, CO, global_cnt):
    c_gather = cost_affinity(H * W, CI, 19)

    c_down_up = cost_conv(H, W, CI, CK, 1) * 2

    c_pixel = cost_conv(H, W, CI, CK, 1)
    c_region = cost_conv(1, global_cnt, CI, CK, 1)

    c_affinity = cost_affinity(H * W, CK, global_cnt)

    c_value = global_cnt * H * W * CK

    c_skip = cost_conv(H, W, CI + CO, CO, 1)
    c_ocr = c_gather + c_down_up + c_pixel + c_region + c_affinity + c_value + c_skip
    return c_ocr


def cost_ppm(H, W, CI, factor=4):
    c_pool = (
        cost_conv(1, 1, CI, CI // factor, 1)
        + cost_conv(2, 2, CI, CI // factor, 1)
        + cost_conv(3, 3, CI, CI // factor, 1)
        + cost_conv(6, 6, CI, CI // factor, 1)
    )

    c_fuse = cost_conv(H, W, CI * 2, CI // factor, 3, padding=1)

    c_ppm = c_pool + c_fuse
    return c_ppm


def cost_aspp(H, W, CI, factor=8):
    c_pool = cost_conv(1, 1, CI, CI // factor, 1)
    c_conv1x1 = cost_conv(H, W, CI, CI // factor, 1)
    c_conv3x3 = (
        cost_conv(H, W, CI, CI // factor, 3, padding=12, dilation=12)
        + cost_conv(H, W, CI, CI // factor, 3, padding=24, dilation=24)
        + cost_conv(H, W, CI, CI // factor, 3, padding=36, dilation=36)
    )

    c_fuse = cost_conv(H, W, 5 * CI // factor, CI // factor, 1)
    c_aspp = c_pool + c_conv1x1 + c_conv3x3 + c_fuse
    return c_aspp


if __name__ == "__main__":
    H, W, C = 128, 128, 512

    ##### Conv 3 x 3
    c_conv = cost_conv(H, W, 2048, 512, 3, padding=1)
    print("GFLOPs of 3x3 conv: {:.1f}".format(c_conv / 1000 ** 3))

    ##### DA
    CI, factor, global_cnt = C, 2, 64
    c_da = cost_double_attention(H, W, CI, factor, global_cnt) + cost_conv(
        H, W, 2048, 512, 3, padding=1
    )
    print("GFLOPs of double-attention: {:.1f}".format(c_da / 1000 ** 3))

    ##### OCR
    CI, CK, CV, CO, global_cnt = C, C // 2, C // 2, C, 19
    c_ocr = cost_ocr(H, W, CI, CK, CV, CO, global_cnt) + cost_conv(
        H, W, 2048, 512, 3, padding=1
    )
    print("GFLOPs of ocr: {:.1f}".format(c_ocr / 1000 ** 3))

    ##### ASPP
    CI, factor = 2048, 8
    c_aspp = cost_aspp(H, W, CI, factor)
    print("GFLOPs of ASPP: {:.1f}".format(c_aspp / 1000 ** 3))

    ##### PPM
    CI, factor = 2048, 4
    c_ppm = cost_ppm(H, W, CI, factor)
    print("GFLOPs of PPM: {:.1f}".format(c_ppm / 1000 ** 3))

    ##### SA
    CI, CK, CV, CO = C, C // 2, C // 2, C
    c_sa = cost_sa(H, W, CI, CK, CV, CO) + cost_conv(H, W, 2048, 512, 3, padding=1)
    print("GFLOPs of SA: {:.1f}".format(c_sa / 1000 ** 3))

    ##### IA
    CI, CK, CV, CO = C, C // 2, C, C
    c_ia = cost_ia(H, W, CI, CK, CV, CO, P=8) + cost_conv(H, W, 2048, 512, 3, padding=1)
    print("GFLOPs of IA: {:.1f}".format(c_ia / 1000 ** 3))

    ##### CC
    CI, CK, CV, CO = 512, 64, 512, 512
    c_cc = cost_rcca(H, W, 2048, CI, CK, CV, CO)
    print("GFLOPs of CC: {:.1f}".format(c_cc / 1000 ** 3))
