#include <iostream>
#include <optional>
#include <vector>
#include <tuple>
#include <bitset>
#include <limits>
#include <iomanip>
#include <fstream>

#define TT_ASSERT(condition, ...)

// #define DEBUG

#ifdef DEBUG
#define DEBUG_PRINT(msg) std::cout << "[DEBUG] " << msg << std::endl;
#else
#define DEBUG_PRINT(msg)
#endif


enum class DataFormat : uint8_t {
    Float32 = 0,
    Float16 = 1,
    Bfp8 = 2,
    Bfp4 = 3,
    Bfp2 = 11,
    Float16_b = 5,
    Bfp8_b = 6,
    Bfp4_b = 7,
    Bfp2_b = 15,
    Lf8 = 10,
    Fp8_e4m3 = 0x1A,
    Int8 = 14,
    Tf32 = 4,
    UInt8 = 30,
    UInt16 = 9,
    Int32 = 8,
    UInt32 = 24,
    RawUInt8 = 0xf0,
    RawUInt16 = 0xf1,
    RawUInt32 = 0xf2,
    Invalid = 0xff
};

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;

inline uint8_t get_max_exp(const std::vector<uint32_t> &vec, bool is_exp_a) {
    uint32_t max = 0;

    for (int i = 0; i < 16; ++i) {
        // mask & shift out exp
        uint32_t exp = (vec[i] & 0x7f800000) >> 23;

        if (is_exp_a) {
            int32_t se = static_cast<int32_t>(exp);
            // need to rebias from 127 to 15
            se = se - 127 + 15;

            if (se > 31) {
                se = 31;
            }
            else if (se < 0) {
                se = 0;
            }

            exp = static_cast<uint32_t>(se);
        }

        if (exp > max) {
            max = exp;
        }
    }
    return max;
}

inline uint32_t get_exp_dword(const std::vector<uint8_t> &vec) {
    TT_ASSERT(vec.size() == 4);

    uint32_t tmp = 0;
    for(int i = 0; i < 4; ++i) {
        tmp = tmp | ((vec[i] & 0xff) << (i*8));
    }
    return tmp;
}

inline uint32_t get_byte(uint32_t word, uint32_t index) {
    TT_ASSERT(index < 4);
    uint32_t mask = 0xff << (8 * index);
    uint32_t masked = word & mask;
    masked = masked >> (8 * index);
    return masked;
}

template <DataFormat BfpFormat, bool truncate_bfp_mantissa=false>
inline uint8_t convert_u32_to_bfp(uint32_t input, uint32_t shared_exp, bool is_exp_a = false) {

    // [DEBUG] MANTISSA_BFP_WIDTH 7
    // [DEBUG] MANTISSA_BFP_SHIFT 17
    // [DEBUG] MANTISSA_BFP_MAX_VAL 127
    // [DEBUG] EXP_MANTISSA_BMSK 7fffffff

    TT_ASSERT(
            BfpFormat == DataFormat::Bfp2   ||
            BfpFormat == DataFormat::Bfp4   ||
            BfpFormat == DataFormat::Bfp8   ||
            BfpFormat == DataFormat::Bfp2_b ||
            BfpFormat == DataFormat::Bfp4_b ||
            BfpFormat == DataFormat::Bfp8_b
            );

    constexpr uint32_t MANTISSA_BFP_WIDTH =
        (BfpFormat == DataFormat::Bfp2 || BfpFormat == DataFormat::Bfp2_b) ? 1 :
        (BfpFormat == DataFormat::Bfp4 || BfpFormat == DataFormat::Bfp4_b) ? 3 : 7;
    constexpr uint32_t MANTISSA_BFP_SHIFT = 24 - MANTISSA_BFP_WIDTH;
    constexpr uint32_t MANTISSA_BFP_MAX_VAL = (1 << MANTISSA_BFP_WIDTH) - 1;

    // DEBUG_PRINT("MANTISSA_BFP_WIDTH " << MANTISSA_BFP_WIDTH);
    // DEBUG_PRINT("MANTISSA_BFP_SHIFT " << MANTISSA_BFP_SHIFT);
    // DEBUG_PRINT("MANTISSA_BFP_MAX_VAL " << MANTISSA_BFP_MAX_VAL);

    //check for both +/- 0.0
    constexpr uint32_t EXP_MANTISSA_BMSK = ((1U << 31) - 1);
    bool is_zero = ((input & EXP_MANTISSA_BMSK) == 0);

    // DEBUG_PRINT("EXP_MANTISSA_BMSK " << std::hex << EXP_MANTISSA_BMSK);
    if (is_zero) {
        return 0;
    }

    uint32_t mantissa = input & 0x007fffff;
    uint32_t exp = (input & 0x7f800000) >> 23;
    uint32_t sign = (input & 0x80000000) >> 31;

    DEBUG_PRINT("\nmantissa(23bit) " << std::bitset<23>(mantissa));
    DEBUG_PRINT("shared exp " << std::bitset<8>(shared_exp) << " exp " << std::bitset<8>(exp));
    if (is_exp_a) {
        int32_t se = static_cast<int32_t>(exp);
        // rebias
        se = se - 127 + 15;
        // check for saturation
        if (se > 31) {
            se = 31;
            mantissa = 0x007fffff;
        }
        else if (se < 0) {
            se = 0;
            mantissa = 0x0;
        }

        exp = static_cast<uint32_t>(se);
    }

    // float mantissa is 23 bits + hidden bit = 24 bits
    // add hidden 1
    mantissa = (1 << 23) | mantissa;
    DEBUG_PRINT("mantissa(24bit) " << std::bitset<24>(mantissa));

    if (shared_exp >= exp) {
        int exp_diff = shared_exp - exp;
        // shift mantissa further down by exp diff
        // In bit-shift operation (A >> B), the result is undefined if B is greater than or equal to the number of bits in A

        DEBUG_PRINT("exp_diff " << exp_diff);
        while (exp_diff > 31) {
            mantissa = mantissa >> 31;
            exp_diff -= 31;
        }
        mantissa = mantissa >> exp_diff;
        DEBUG_PRINT("mantissa(shifting)\t" << std::bitset<32>(mantissa));
    }

    // this needs to become 3 bits so shift 21 times
    if (truncate_bfp_mantissa) {
        // Truncation: Round down
        mantissa = mantissa >> MANTISSA_BFP_SHIFT;
    }
    else {
        // Round mantissa to nearest even
        mantissa += 1 << (MANTISSA_BFP_SHIFT-1);

        DEBUG_PRINT("mantissa(rounding)\t" << std::bitset<32>(mantissa));
        mantissa = mantissa >> MANTISSA_BFP_SHIFT;
        DEBUG_PRINT("mantissa(shifting)\t" << std::bitset<32>(mantissa));
        if(mantissa > MANTISSA_BFP_MAX_VAL) mantissa = MANTISSA_BFP_MAX_VAL;
        DEBUG_PRINT("mantissa(max?)\t\t" << std::bitset<32>(mantissa));
    }

    // add sign bit only if result is not 0
    if (0 == mantissa) {
        sign = 0;
    }
    mantissa = (sign << MANTISSA_BFP_WIDTH) | mantissa;
    return mantissa;
}

inline uint32_t convert_bfp_to_u32(DataFormat bfp_format, uint8_t data, uint8_t shared_exp, bool is_exp_a = false) {
    uint32_t exp = shared_exp;
    uint32_t out_num = 0;
    if ((bfp_format == DataFormat::Bfp2_b) || (bfp_format == DataFormat::Bfp2)) {
        uint32_t sign = data >> 1;
        uint32_t man = data & 0x1;

        // Shift mantissa up until there is a 1 in bit 1
        int shift_cnt = 0;
        if (man == 0) {
            man = 0;
            exp = 0;
        } else {
            // shift again to put first non-hidden mantissa
            // bit in bit 1
            man = man << 1;
            man = man & 0x1;

            // adjust exponent
            TT_ASSERT(exp >= (uint32_t)shift_cnt, "incorrect shift_cnt");
            exp = exp - shift_cnt;

            // if exp_a rebias exp to 127
            if (is_exp_a) {
                exp = exp - 15 + 127;
            }
        }

        // put s, e, m together
        out_num = (sign << 31) | (exp << 23) | (man << 22);
    } else if ((bfp_format == DataFormat::Bfp4_b) || (bfp_format == DataFormat::Bfp4)) {
        uint32_t sign = data >> 3;
        uint32_t man = data & 0x7;

        // Shift mantissa up until there is a 1 in bit 3
        int shift_cnt = 0;
        if (man == 0) {
            man = 0;
            exp = 0;
        } else {
            while ((man & 0x04) == 0) {
                man = man << 1;
                shift_cnt++;
            }
            // shift one more time and zero the
            // hidden top mantissa bit
            // shift again to put first non-hidden mantissa
            // bit in bit 3
            man = man << 1;
            man = man & 0x7;

            // adjust exponent
            TT_ASSERT(exp >= (uint32_t)shift_cnt, "incorrect shift_cnt");
            exp = exp - shift_cnt;

            // if exp_a rebias exp to 127
            if (is_exp_a) {
                exp = exp - 15 + 127;
            }
        }

        // put s, e, m together
        out_num = (sign << 31) | (exp << 23) | (man << 20);
    } else if ((bfp_format == DataFormat::Bfp8_b) || (bfp_format == DataFormat::Bfp8)) {
        uint32_t sign = data >> 7;
        uint32_t man = data & 0x7f;

        // Shift mantissa up until there is a 1 in bit 6
        int shift_cnt = 0;
        if (man == 0) {
            man = 0;
            exp = 0;
            DEBUG_PRINT("man == 0");
        } else {
            // shift_cnt = 6 - (31 - __builtin_clz(man));
            // man = (man << (shift_cnt + 1)) & 0x7f;
            while ((man & 0x40) == 0) {
                man = man << 1;
                shift_cnt++;
            }
            // shift one more time and zero the
            // hidden top mantissa bit
            // shift again to put first non-hidden mantissa
            // bit in bit 7
            man = man << 1;
            man = man & 0x7f;

            // adjust exponent
            TT_ASSERT(exp >= (uint32_t)shift_cnt, "incorrect shift_cnt");
            exp = exp - shift_cnt;

            // if exp_a rebias exp to 127
            if (is_exp_a) {
                exp = exp - 15 + 127;
            }
        }

        // put s, e, m together
        out_num = (sign << 31) | (exp << 23) | (man << 16);
    }
    return out_num;
}

template <DataFormat BfpFormat>
inline uint32_t create_packed_bfp_packed_as_u32(const std::vector<uint32_t> &u32_vec, uint32_t shared_exp, bool is_exp_a) {
    TT_ASSERT(
            BfpFormat == DataFormat::Bfp2   ||
            BfpFormat == DataFormat::Bfp4   ||
            BfpFormat == DataFormat::Bfp8   ||
            BfpFormat == DataFormat::Bfp2_b ||
            BfpFormat == DataFormat::Bfp4_b ||
            BfpFormat == DataFormat::Bfp8_b
            );
    constexpr int nums_in_dword =
        (BfpFormat == DataFormat::Bfp2 || BfpFormat == DataFormat::Bfp2_b) ? 16 :
        (BfpFormat == DataFormat::Bfp4 || BfpFormat == DataFormat::Bfp4_b) ? 8 : 4;

    uint32_t tmp_o = 0;
    uint32_t mask = (1 << (32 / nums_in_dword)) - 1;
    for (int i = nums_in_dword - 1; i >= 0; --i) // [0] in LSBs of dword
    {
        uint32_t conv_num = convert_u32_to_bfp<BfpFormat, false>(u32_vec[i], shared_exp, is_exp_a);
        tmp_o = tmp_o << (32 / nums_in_dword);
        tmp_o = tmp_o | (conv_num & mask);
    }
    return tmp_o;
}

template <DataFormat BfpFormat>
inline std::vector<uint32_t> pack_fp32_vec_as_bfp_tiles(const std::vector<float> &fp32_vec, bool row_major_input, bool is_exp_a) {

    TT_ASSERT(
            BfpFormat == DataFormat::Bfp2   ||
            BfpFormat == DataFormat::Bfp4   ||
            BfpFormat == DataFormat::Bfp8   ||
            BfpFormat == DataFormat::Bfp2_b ||
            BfpFormat == DataFormat::Bfp4_b ||
            BfpFormat == DataFormat::Bfp8_b
            );

    auto tile_H = TILE_HEIGHT;
    auto tile_W = TILE_WIDTH;
    auto face_H = FACE_HEIGHT;
    auto face_W = FACE_WIDTH;
    auto tile_HW = tile_H * tile_W;
    auto subtiles_in_tile_row = tile_H / face_H;
    auto subtiles_in_tile_col = tile_W / face_W;
    auto subtile_rows = face_H;
    auto subtile_cols = face_W;

    int num_float_in_tile = tile_HW;
    TT_ASSERT(fp32_vec.size() % num_float_in_tile == 0);
    uint32_t num_tiles = fp32_vec.size() / num_float_in_tile;

    std::vector<uint32_t> packed_result;

    std::vector<uint8_t> exponents;
    std::vector<uint32_t> data;

    int num_exponents_in_dword = 4;
    int num_mantissas_in_dword =
        (BfpFormat == DataFormat::Bfp2 || BfpFormat == DataFormat::Bfp2_b) ? 16 :
        (BfpFormat == DataFormat::Bfp4 || BfpFormat == DataFormat::Bfp4_b) ? 8 : 4;
    int fp32_element_index = 0;
    for (uint32_t tile_index = 0; tile_index < num_tiles; ++tile_index) {
        std::vector<uint32_t> packed_data;
        for (uint32_t tr = 0; tr < subtiles_in_tile_row; ++tr) {
            for (uint32_t tc = 0; tc < subtiles_in_tile_col; ++tc) {
                for (uint32_t i = 0; i < subtile_rows; ++i) {
                    std::vector<uint32_t> single_row;
                    // populate a single row
                    for (uint32_t j = 0; j < subtile_cols; ++j) {
                        int data_index;
                        if (row_major_input) {
                            data_index = (tr*face_H + i)*tile_W + (tc*face_W + j) + (num_float_in_tile * tile_index);
                        } else {
                            data_index = fp32_element_index++;
                        }
                        float float_num = fp32_vec.at(data_index);
                        uint32_t uint32_num = *reinterpret_cast<uint32_t*>(&float_num);
                        single_row.push_back(uint32_num);
                    }

                    uint8_t exp = get_max_exp(single_row, is_exp_a);
                    exponents.push_back(exp);

                    if (exponents.size() % num_exponents_in_dword == 0) {
                        packed_result.push_back(get_exp_dword(exponents));
                        exponents.clear();
                    }

                    for (uint32_t u32_datum : single_row) {
                        data.push_back(u32_datum);
                        if (data.size() % num_mantissas_in_dword == 0) {
                            uint32_t datum = create_packed_bfp_packed_as_u32<BfpFormat>(data, exp, is_exp_a);
                            packed_data.push_back(datum);
                            data.clear();
                        }
                    }
                }
            }
        }
        // prepend exponents to follow data packing order:
        //  16 exponents for sub-tile 0​
        //      exp_row0, exp_row1, … exp_row15​
        //  16 exponents for sub-tile 1​
        //  16 exponents for sub-tile 2​
        //  16 exponents for sub-tile 3​
        //  entire sub-tile 0 (RM layout)​
        //  entire sub-tile 1 (RM layout)​
        //  entire sub-tile 2 (RM layout)​
        //  entire sub-tile 3 (RM layout)
        packed_result.insert(packed_result.end(), packed_data.begin(), packed_data.end());
    }

    return packed_result;
}

template <DataFormat BfpFormat>
inline std::tuple<std::vector<uint8_t>, std::vector<uint32_t>> pack_fp32_vec_as_bfp_one_block(const std::vector<float> &fp32_vec, bool is_exp_a=false) {
    TT_ASSERT(
            BfpFormat == DataFormat::Bfp2   ||
            BfpFormat == DataFormat::Bfp4   ||
            BfpFormat == DataFormat::Bfp8   ||
            BfpFormat == DataFormat::Bfp2_b ||
            BfpFormat == DataFormat::Bfp4_b ||
            BfpFormat == DataFormat::Bfp8_b
            );

    auto face_W = FACE_WIDTH;
    auto subtile_cols = face_W;

    std::vector<uint32_t> packed_result;

    std::vector<uint8_t> exponents;
    std::vector<uint32_t> data;

    int num_mantissas_in_dword =
        (BfpFormat == DataFormat::Bfp2 || BfpFormat == DataFormat::Bfp2_b) ? 16 :
        (BfpFormat == DataFormat::Bfp4 || BfpFormat == DataFormat::Bfp4_b) ? 8 : 4;
    int fp32_element_index = 0;

    std::vector<uint32_t> single_row;
    // populate a single row
    for (uint32_t j = 0; j < subtile_cols; ++j) {
        int data_index = fp32_element_index++;
        float float_num = fp32_vec.at(data_index);
        uint32_t uint32_num = *reinterpret_cast<uint32_t*>(&float_num);
        single_row.push_back(uint32_num);
    }

    uint8_t exp = get_max_exp(single_row, is_exp_a);
    exponents.push_back(exp);

    for (uint32_t u32_datum : single_row) {
        data.push_back(u32_datum);
        if (data.size() % num_mantissas_in_dword == 0) {
            uint32_t datum = create_packed_bfp_packed_as_u32<BfpFormat>(data, exp, is_exp_a);
            packed_result.push_back(datum);
            data.clear();
        }
    }

    return {exponents, packed_result};
}

void clearLower16Bits(std::vector<float>& block) {
    for (auto& value : block) {
        // float 값을 uint32_t로 reinterpret
        uint32_t* int_ptr = reinterpret_cast<uint32_t*>(&value);

        // 하위 16비트를 0으로 설정
        *int_ptr &= 0xFFFF0000;
    }
}


// grad is converted to zero!
void adamw_bfp8b_zero_grad_case() {
    std::vector<float> block { 0.0339,  0.0339,  0.0339,  0.0339,  0.0339,  0.0275,  0.0008, -0.0210,
        -0.0674, -0.0991, -0.1128, -0.1270, -0.0496,  0.0004,  0.0359,  0.0471};

    clearLower16Bits(block);

    auto [exponent, packed_data] = pack_fp32_vec_as_bfp_one_block<DataFormat::Bfp8_b>(block);

    std::cout << "shard exponent " << (uint32_t)exponent[0] << " : " << std::hex << (uint32_t)exponent[0] << "\n";

    for (auto data : packed_data) {
        // 8비트 값 4개를 저장할 벡터
        std::vector<uint8_t> bytes(4);

        // 각 바이트를 분리 (Big Endian 순서로 분리)
        for (int i = 0; i < 4; ++i) {
            bytes[i] = (data >> (i * 8)) & 0xFF;
        }

        // 각 바이트의 sign bit(1-bit)와 mantissa(7-bit)를 출력
        for (int i = 0; i < 4; ++i) {
            uint8_t byte = bytes[i];
            bool sign = (byte >> 7) & 0x1; // MSB (Most Significant Bit)
            uint8_t mantissa = byte & 0x7F; // 나머지 7비트 (LSB 7 bits)

            std::cout << "Byte " << i << ": Sign = " << sign
                << ", Mantissa = " << static_cast<int>(mantissa) << "\n";
        }
    }

    for (auto data : packed_data) {
        std::vector<uint8_t> bytes(4);
        for (int i = 0; i < 4; ++i) {
            bytes[i] = (data >> (i * 8)) & 0xFF;
            auto u32_val = convert_bfp_to_u32(DataFormat::Bfp8_b, bytes[i], exponent[0]);
            float bf16_val = *reinterpret_cast<float*>(&u32_val);

            std::cout << bf16_val << "\n";
        }
    }
}

std::vector<float> read_values_from_file(const std::string& filename, float& default_value) {
    std::ifstream file(filename);
    std::vector<float> values;

    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return values;
    }

    file >> default_value;

    float val;
    while (file >> val && values.size() < 16) {
        values.push_back(val);
    }

    return values;
}

// What is this case for?
void convert_u32_to_bfp_case() {
    uint8_t out1 =  convert_u32_to_bfp<DataFormat::Bfp8_b>(0xff800000, 0xff);
    uint8_t out2 =  convert_u32_to_bfp<DataFormat::Bfp8_b>(0xff7f0000, 0xff);
    std::cout << std::hex << (uint32_t)out1 << " " << (uint32_t)out2 << std::endl;
}

int main(void)
{
    float default_value = 0.0f;
    std::vector<float> block = read_values_from_file("data.txt", default_value);

    if (block.empty()) {
        std::cerr << "Error: No values found, using default value instead.\n";
        block.assign(16, default_value);
    } else {
        while (block.size() < 16) {
            block.push_back(default_value);
        }
    }
    clearLower16Bits(block);

    std::cout << "16 bfloat16 values before being packed into bfp8\n";
    std::cout << "Bfloat16 Value\tBit pattern\n";
    for(auto data : block) {
        auto u32_val = *reinterpret_cast<uint32_t*>(&data);
        std::cout << std::setw(14) << data << "\t" <<std::bitset<32>(u32_val) << "\n";
    }
    std::cout << "\nPacking into bfp8\n";
    auto [exponent, packed_data] = pack_fp32_vec_as_bfp_one_block<DataFormat::Bfp8_b>(block);

    std::cout << "Shared exponent : " << static_cast<uint32_t>(exponent[0]) << "\n";
    std::cout << "Sign + Mantissa : ";
    for (auto data : packed_data) {
        std::vector<uint8_t> bytes(4);
        for (int i = 0; i < 4; ++i) {
            bytes[i] = (data >> (i * 8)) & 0xFF;
            std::cout << std::hex << static_cast<uint32_t>(bytes[i]) << " ";
        }
    }
    std::cout << "\n\n";

    std::cout << "16 bfloat16 values after unpacking from bfp8\n";
    std::cout << "Bfloat16 Value\tBit pattern\n";
    for (auto data : packed_data) {
        std::vector<uint8_t> bytes(4);
        for (int i = 0; i < 4; ++i) {
            bytes[i] = (data >> (i * 8)) & 0xFF;
            auto u32_val = convert_bfp_to_u32(DataFormat::Bfp8_b, bytes[i], exponent[0]);
            float bf16_val = *reinterpret_cast<float*>(&u32_val);

            auto u16_val = *reinterpret_cast<uint32_t*>(&bf16_val);
            std::cout << std::setw(14) << bf16_val << "\t" << std::bitset<32>(u16_val) << "\n";
        }
    }

    return 0;
}
