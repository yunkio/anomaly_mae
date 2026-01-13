# Input Projection과 Positional Encoding 분석

## 질문 요약

1. Input projection이 다변량 시계열을 하나의 차원으로 만드는 과정인가?
2. 시계열의 시간적 순서는 잘 유지되고 있나?
3. 각 feature의 시간적 정보를 잃는 것은 아닌가?
4. Positional encoding은 제대로 작동하고 있나?

---

## (1) Input Projection - 무엇을 하는가?

### Token-Level Mode의 Input Projection

```python
# 코드 (line 348)
self.input_projection = nn.Linear(config.num_features, config.d_model)
# num_features=5 -> d_model=64

# Forward (line 526-527)
x_embed = self.input_projection(x)  # (batch, seq, d_model)
x_embed = x_embed.transpose(0, 1)   # (seq, batch, d_model)
```

### 구체적인 예시

입력 데이터:
```
x: (batch=2, seq_length=100, num_features=5)

시간 스텝별 데이터 (하나의 sample):
t=0:  [CPU=0.3, Memory=0.5, Disk=0.2, Network=0.4, Response=0.35]
t=1:  [CPU=0.32, Memory=0.51, Disk=0.21, Network=0.41, Response=0.36]
t=2:  [CPU=0.35, Memory=0.52, Disk=0.22, Network=0.39, Response=0.38]
...
t=99: [CPU=0.28, Memory=0.48, Disk=0.19, Network=0.42, Response=0.33]
```

Input projection 적용 **각 시간 스텝에 독립적으로**:
```python
# t=0의 변환
x[t=0] = [0.3, 0.5, 0.2, 0.4, 0.35]  # shape: (5,)
         ↓ Linear(5 -> 64)
embed[t=0] = [e0, e1, e2, ..., e63]  # shape: (64,)

# t=1의 변환
x[t=1] = [0.32, 0.51, 0.21, 0.41, 0.36]  # shape: (5,)
         ↓ Linear(5 -> 64)
embed[t=1] = [e'0, e'1, e'2, ..., e'63]  # shape: (64,)

# ...모든 시간 스텝에 대해 반복
```

최종 결과:
```
x_embed: (batch=2, seq_length=100, d_model=64)

하나의 sample:
embed[t=0]:  [64차원 벡터]  <- t=0의 5개 feature를 표현
embed[t=1]:  [64차원 벡터]  <- t=1의 5개 feature를 표현
embed[t=2]:  [64차원 벡터]  <- t=2의 5개 feature를 표현
...
embed[t=99]: [64차원 벡터]  <- t=99의 5개 feature를 표현
```

### Patch-Level Mode의 Input Projection

```python
# Patchify (line 523)
x = x.reshape(batch_size, self.num_patches, self.patch_size * num_features)
# (batch, 100, 5) -> (batch, 10, 50)

# Patch embed (line 344)
self.patch_embed = nn.Linear(config.patch_size * config.num_features, config.d_model)
# (patch_size * num_features = 10 * 5 = 50) -> d_model=64
```

구체적인 예시:
```
원본 데이터: (batch, seq_length=100, num_features=5)

Patchify 후:
Patch 0: t=0~9의 모든 feature
  = [CPU[0], Mem[0], Disk[0], Net[0], Resp[0],
     CPU[1], Mem[1], Disk[1], Net[1], Resp[1],
     ...
     CPU[9], Mem[9], Disk[9], Net[9], Resp[9]]
  = 10 time steps × 5 features = 50 values

Patch 1: t=10~19의 모든 feature
  = [CPU[10], Mem[10], ..., Resp[19]]
  = 50 values

...

Patch 9: t=90~99의 모든 feature
  = 50 values

각 patch를 embedding:
Patch 0 (50,) -> Linear(50 -> 64) -> embed[0] (64,)
Patch 1 (50,) -> Linear(50 -> 64) -> embed[1] (64,)
...
Patch 9 (50,) -> Linear(50 -> 64) -> embed[9] (64,)

최종: (batch, num_patches=10, d_model=64)
```

---

## (2) 시간적 순서는 유지되는가?

### ✅ **네, 완벽하게 유지됩니다!**

### Token-Level Mode

```
입력 순서:
x[0, :, :] = t=0의 5개 feature
x[1, :, :] = t=1의 5개 feature
x[2, :, :] = t=2의 5개 feature
...
x[99, :, :] = t=99의 5개 feature

Embedding 후:
embed[0, :, :] = t=0의 embedding
embed[1, :, :] = t=1의 embedding
embed[2, :, :] = t=2의 embedding
...
embed[99, :, :] = t=99의 embedding
```

**시간 축(첫 번째 차원)은 그대로 보존됩니다.**

Input projection은 **각 시간 스텝을 독립적으로** (5 -> 64) 변환할 뿐이지, 시간 스텝의 순서를 바꾸지 않습니다.

### Patch-Level Mode

```
입력 순서:
Patch 0: t=0~9
Patch 1: t=10~19
Patch 2: t=20~29
...
Patch 9: t=90~99

Embedding 후:
embed[0] = Patch 0 (t=0~9) embedding
embed[1] = Patch 1 (t=10~19) embedding
embed[2] = Patch 2 (t=20~29) embedding
...
embed[9] = Patch 9 (t=90~99) embedding
```

**Patch 순서 = 시간 순서로 유지됩니다.**

---

## (3) 각 Feature의 시간적 정보를 잃는가?

### ❌ **아니요, 잃지 않습니다!**

중요한 오해를 바로잡겠습니다:

### Token-Level Mode

```python
# 잘못된 이해:
# "5개 feature를 64차원으로 합쳐서 feature별 정보를 잃는다?"

# 실제:
# Linear projection은 학습 가능한 변환입니다!
self.input_projection = nn.Linear(5, 64)

# 파라미터 행렬:
W: (5, 64)
b: (64,)

# 변환:
output = input @ W + b
# input: (5,)
# output: (64,) = [w1*CPU + w2*Mem + w3*Disk + w4*Net + w5*Resp + b1,
#                  w6*CPU + w7*Mem + ...,
#                  ...]
```

**이것은 정보 손실이 아니라 정보 확장입니다!**

- 입력: 5차원 (5개 feature)
- 출력: 64차원 (더 풍부한 표현 공간)

각 feature의 정보는:
1. **보존됨**: 64개 차원에 분산되어 인코딩됨
2. **결합됨**: Feature 간 상호작용이 학습됨
3. **확장됨**: 더 복잡한 패턴을 표현할 수 있는 고차원 공간으로 매핑

### 시간적 정보 보존 메커니즘

```
시간 스텝마다 별도로 embedding되므로:

t=0의 CPU=0.3  ──┐
t=0의 Memory=0.5 ├──> Linear ──> embed[t=0] (64-dim)
t=0의 Disk=0.2   ├──>
...              ─┘

t=1의 CPU=0.32  ──┐
t=1의 Memory=0.51 ├──> Linear ──> embed[t=1] (64-dim)
...              ─┘

...

t=99의 CPU=0.28  ──┐
...               ├──> Linear ──> embed[t=99] (64-dim)
                  ─┘
```

각 시간 스텝의 embedding은 **그 시간 스텝의 feature snapshot**을 표현합니다.

시간 스텝 간의 관계는 이후 **Transformer의 attention mechanism**에서 학습됩니다!

### Patch-Level Mode는?

```
Patch 0 (t=0~9):
  [CPU[0], Mem[0], Disk[0], Net[0], Resp[0],
   CPU[1], Mem[1], Disk[1], Net[1], Resp[1],
   ...
   CPU[9], Mem[9], Disk[9], Net[9], Resp[9]]
  = 50 values
  ↓ Linear(50 -> 64)
  embed[0] = 64-dim vector

이 64차원 벡터는:
- t=0~9의 모든 시간 스텝 정보 포함
- 5개 feature의 시간적 변화 패턴 포함
- 더 풍부한 시공간 정보를 담음
```

**정보 손실이 아니라 정보 압축입니다.**
- 입력: 50차원
- 출력: 64차원
- 실제로는 차원이 증가했지만, 10개 시간 스텝의 패턴을 하나의 표현으로 요약

---

## (4) Positional Encoding은 작동하는가?

### ✅ **네, 제대로 작동합니다!**

### 코드 확인

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 짝수 차원
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 홀수 차원

        self.register_buffer('pe', pe)  # 학습되지 않는 고정 파라미터

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]  # 위치 인코딩을 더함
        return x
```

### 동작 과정

#### Token-Level Mode (seq_len=100)

```python
# Forward에서 (line 544)
x_masked = self.pos_encoder(x_masked)

# 실제 동작:
x_masked: (100, batch, 64)
pe: (5000, 1, 64) 중에서 [:100, :, :] 사용

# 각 위치별로:
x_masked[0] = x_masked[0] + pe[0]   # position 0 encoding 추가
x_masked[1] = x_masked[1] + pe[1]   # position 1 encoding 추가
x_masked[2] = x_masked[2] + pe[2]   # position 2 encoding 추가
...
x_masked[99] = x_masked[99] + pe[99] # position 99 encoding 추가
```

**각 시간 스텝에 고유한 위치 정보가 추가됩니다!**

#### Patch-Level Mode (seq_len=10)

```python
x_masked: (10, batch, 64)
pe: (5000, 1, 64) 중에서 [:10, :, :] 사용

x_masked[0] = x_masked[0] + pe[0]   # Patch 0 (t=0~9) 위치 정보
x_masked[1] = x_masked[1] + pe[1]   # Patch 1 (t=10~19) 위치 정보
...
x_masked[9] = x_masked[9] + pe[9]   # Patch 9 (t=90~99) 위치 정보
```

**각 patch에 순서 정보가 추가됩니다.**

### Positional Encoding의 수식

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

예시 (d_model=64):
pos=0:
  PE[0, 0] = sin(0 / 10000^(0/64)) = sin(0) = 0
  PE[0, 1] = cos(0 / 10000^(0/64)) = cos(0) = 1
  PE[0, 2] = sin(0 / 10000^(2/64)) = 0
  PE[0, 3] = cos(0 / 10000^(2/64)) = 1
  ...

pos=1:
  PE[1, 0] = sin(1 / 10000^(0/64)) = sin(1) ≈ 0.841
  PE[1, 1] = cos(1 / 10000^(0/64)) = cos(1) ≈ 0.540
  PE[1, 2] = sin(1 / 10000^(2/64)) ≈ 0.9999
  PE[1, 3] = cos(1 / 10000^(2/64)) ≈ 0.999
  ...
```

**각 위치는 고유한 64차원 벡터를 가집니다.**

---

## 종합 정리

### ✅ 시간적 정보 보존 메커니즘

1. **Input Projection**:
   - Feature 정보를 저차원(5)에서 고차원(64)으로 확장
   - 각 시간 스텝을 독립적으로 변환
   - 시간 순서는 그대로 유지

2. **Positional Encoding**:
   - 각 시간 스텝/패치에 고유한 위치 정보 추가
   - Transformer가 순서를 인식할 수 있게 함
   - 절대 위치 정보 제공

3. **Transformer Attention**:
   - 모든 시간 스텝 간의 관계를 학습
   - 시간적 의존성을 모델링
   - Self-attention으로 장거리 의존성 포착

### 정보 흐름 예시

```
입력:
t=0: [CPU=0.3, Mem=0.5, Disk=0.2, Net=0.4, Resp=0.35]
t=1: [CPU=0.32, Mem=0.51, Disk=0.21, Net=0.41, Resp=0.36]
...

↓ Input Projection (5 -> 64)

t=0: [64-dim embedding of features at t=0]
t=1: [64-dim embedding of features at t=1]
...

↓ Positional Encoding

t=0: [64-dim embedding] + [PE for position 0]
t=1: [64-dim embedding] + [PE for position 1]
...

↓ Transformer Encoder (attention)

각 시간 스텝이 다른 모든 시간 스텝과 상호작용
→ 시간적 패턴 학습
→ Feature 간 상호작용 학습
→ 장거리 의존성 포착

↓ Output

재구성된 시계열
```

---

## 잠재적 개선 방향 (현재는 문제 없음)

현재 구현은 **정상적으로 작동**하지만, 이론적으로 개선 가능한 부분:

### 1. Feature별 독립적인 Embedding (선택적)

```python
# 현재: 모든 feature를 함께 embed
self.input_projection = nn.Linear(5, 64)

# 대안: Feature별로 독립적으로 embed 후 결합
self.feature_embeds = nn.ModuleList([
    nn.Linear(1, 12) for _ in range(5)  # 각 feature -> 12 dim
])
# 총 5 * 12 = 60 dim, 나머지 4dim은 feature 간 상호작용용
```

하지만 현재 방식도 충분히 효과적입니다.

### 2. Learnable Positional Encoding (선택적)

```python
# 현재: 고정된 sinusoidal encoding
self.register_buffer('pe', pe)

# 대안: 학습 가능한 positional encoding
self.pos_embed = nn.Parameter(torch.randn(max_len, 1, d_model))
```

하지만 sinusoidal은 일반화 성능이 더 좋습니다.

---

## 결론

**모든 질문에 대한 답:**

1. **Input projection이 다변량을 하나로 만드나?**
   - ✅ 네, 하지만 정보 **손실** 없이 고차원 공간으로 **확장**합니다.

2. **시간적 순서는 유지되나?**
   - ✅ 완벽하게 유지됩니다. 각 시간 스텝의 위치는 변하지 않습니다.

3. **Feature의 시간적 정보를 잃나?**
   - ❌ 아니요. 각 시간 스텝마다 embedding되므로 시간 정보 보존됩니다.
   - Positional encoding이 추가로 위치 정보를 명시적으로 제공합니다.
   - Transformer attention이 시간적 관계를 학습합니다.

4. **Positional encoding은 작동하나?**
   - ✅ 네, 제대로 작동합니다. 각 위치에 고유한 인코딩을 추가합니다.

**현재 구현은 이론적으로 정확하고 실무적으로 효과적입니다!**
