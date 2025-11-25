### Benchmark Results

Benchmarking the best performing kernels and comparing against cublasLT. 

### Ampere Architecture
* Numbers are of a RTX 4060Ti

<table>
  <thead>
    <tr>
      <th rowspan="2">Size</th>
      <th colspan="2">FP32 (TF32 accum)</th>
    </tr>
    <tr>
      <th>Best Implementation</th>
      <th>cublasLT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>128 × 128 × 128</td>
      <td>330.862 GFlops</td>
      <td>641.252 GFlops</td>
    </tr>
    <tr>
      <td>256 × 256 × 256</td>
      <td>1.598 TFlops</td>
      <td>3.857 TFlops</td>
    </tr>
    <tr>
      <td>512 × 512 × 512</td>
      <td>7.210 TFlops</td>
      <td>13.029 TFlops</td>
    </tr>
    <tr>
      <td>1024 × 1024 × 1024</td>
      <td>15.415 TFlops</td>
      <td>19.883 TFlops</td>
    </tr>
    <tr>
      <td>2048 × 2048 × 2048</td>
      <td>17.071 TFlops</td>
      <td>21.173 TFlops</td>
    </tr>
    <tr>
      <td>4096 × 4096 × 4096</td>
      <td>18.281 TFlops</td>
      <td>22.342 TFlops</td>
    </tr>
    <tr>
      <td>8192 × 8192 × 8192</td>
      <td>18.850 TFlops</td>
      <td>23.384 TFlops</td>
    </tr>
  </tbody>
</table>
