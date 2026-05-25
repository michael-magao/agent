// Package tools provides OpenClaw tool implementations.
package tools

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// AmazonProductItem 亚马逊商品条目
type AmazonProductItem struct {
	ASIN         string  `json:"asin"`
	Title        string  `json:"title"`
	Brand        string  `json:"brand"`
	Price        float64 `json:"price"`
	Rating       float64 `json:"rating"`
	ReviewsCount int     `json:"reviews_count"`
}

// AmazonSearchResult 搜索结果
type AmazonSearchResult struct {
	Keyword      string              `json:"keyword"`
	TotalResults int                 `json:"total_results"`
	Items        []AmazonProductItem `json:"items"`
}

// ProductDevReport 产品开发建议报告（供 LLM 分析后的结构化输出）
type ProductDevReport struct {
	MarketOverview     string `json:"market_overview"`     // 市场概览
	CompetitorAnalysis string `json:"competitor_analysis"` // 竞品分析
	UserPainPoints     string `json:"user_pain_points"`    // 用户痛点
	Recommendations    string `json:"recommendations"`     // 产品开发建议
	ROIAssessment      string `json:"roi_assessment"`      // 投入产出评估
}

// AmazonClient 封装 PA-API 调用（Go 侧可通过 Python 或 HTTP 代理复用实现）
type AmazonClient struct {
	AccessKey  string
	SecretKey  string
	PartnerTag string
	Country    string
}

// SearchProducts 搜索亚马逊商品
func (c *AmazonClient) SearchProducts(keyword string) (*AmazonSearchResult, error) {
	// 优先调用 Python 工具（同仓库内）
	if data, err := c.callPythonTool("amazon_search_products", keyword); err == nil {
		var r AmazonSearchResult
		if json.Unmarshal([]byte(data), &r) == nil {
			return &r, nil
		}
	}
	// 无 Python 时返回 mock
	return mockSearchResult(keyword), nil
}

// AnalyzeForProductDev 分析并返回产品开发建议摘要
func (c *AmazonClient) AnalyzeForProductDev(keyword string) (string, error) {
	if out, err := c.callPythonTool("amazon_analyze_for_product_dev", keyword); err == nil {
		return out, nil
	}
	r := mockSearchResult(keyword)
	return formatAnalysisPrompt(r), nil
}

func (c *AmazonClient) callPythonTool(toolName, keyword string) (string, error) {
	root := findProjectRoot()
	script := fmt.Sprintf(
		`import sys; sys.path.insert(0, %q)
from pkg.agentic.tools.amazon_product import amazon_search_products, amazon_analyze_for_product_dev
tool_name = sys.argv[1]
kw = sys.argv[2]
print(amazon_search_products(kw, "US") if tool_name == "amazon_search_products" else amazon_analyze_for_product_dev(kw, "US"))`,
		root)

	cmd := exec.Command(pythonExecutable(), "-c", strings.TrimSpace(script), toolName, keyword)
	cmd.Dir = root
	cmd.Env = append(os.Environ(),
		"AMAZON_PAAPI_ACCESS_KEY="+c.AccessKey,
		"AMAZON_PAAPI_SECRET_KEY="+c.SecretKey,
		"AMAZON_PAAPI_PARTNER_TAG="+c.PartnerTag,
		"AMAZON_PAAPI_COUNTRY="+c.Country,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("%w: %s", err, out)
	}
	return string(out), nil
}

func pythonExecutable() string {
	if path := os.Getenv("PYTHON_BIN"); path != "" {
		return path
	}
	if path, err := exec.LookPath("python3"); err == nil {
		return path
	}
	return "python"
}

func findProjectRoot() string {
	wd, _ := os.Getwd()
	for d := wd; d != "" && d != "/"; d = filepath.Dir(d) {
		if _, err := os.Stat(filepath.Join(d, "pkg", "agentic", "tools", "amazon_product.py")); err == nil {
			return d
		}
	}
	return wd
}

func mockSearchResult(keyword string) *AmazonSearchResult {
	return &AmazonSearchResult{
		Keyword:      keyword,
		TotalResults: 3,
		Items: []AmazonProductItem{
			{ASIN: "B0MOCK01", Title: "Example " + keyword + " Product A", Brand: "BrandA", Price: 29.99, Rating: 4.5, ReviewsCount: 1234},
			{ASIN: "B0MOCK02", Title: "Example " + keyword + " Product B", Brand: "BrandB", Price: 19.99, Rating: 4.2, ReviewsCount: 890},
			{ASIN: "B0MOCK03", Title: "Example " + keyword + " Product C", Brand: "BrandC", Price: 39.99, Rating: 4.7, ReviewsCount: 2100},
		},
	}
}

func formatAnalysisPrompt(r *AmazonSearchResult) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("## 关键词: %s\n", r.Keyword))
	b.WriteString(fmt.Sprintf("## 检索商品数: %d\n\n### 竞品样本\n", r.TotalResults))
	for i, it := range r.Items {
		b.WriteString(fmt.Sprintf("%d. [%s] 品牌:%s 价格:$%.2f 评分:%.1f 评论数:%d\n",
			i+1, it.Title, it.Brand, it.Price, it.Rating, it.ReviewsCount))
	}
	b.WriteString("\n---\n请基于以上数据生成《产品开发投入建议报告》：\n")
	b.WriteString("1. 市场概览 2. 竞品分析 3. 用户痛点 4. 产品开发建议 5. 投入产出评估\n")
	return b.String()
}

// NewAmazonClientFromEnv 从环境变量创建客户端
func NewAmazonClientFromEnv() *AmazonClient {
	return &AmazonClient{
		AccessKey:  os.Getenv("AMAZON_PAAPI_ACCESS_KEY"),
		SecretKey:  os.Getenv("AMAZON_PAAPI_SECRET_KEY"),
		PartnerTag: os.Getenv("AMAZON_PAAPI_PARTNER_TAG"),
		Country:    getEnv("AMAZON_PAAPI_COUNTRY", "US"),
	}
}

func getEnv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

// FetchViaPAAPI 直接调用 PA-API（需实现 AWS SigV4 签名，此处预留接口）
// 完整实现可参考: https://webservices.amazon.com/paapi5/documentation/sending-request.html
func FetchViaPAAPI(keyword string, client *AmazonClient) (*http.Response, error) {
	if client.AccessKey == "" || client.SecretKey == "" {
		return nil, fmt.Errorf("AMAZON_PAAPI_ACCESS_KEY and AMAZON_PAAPI_SECRET_KEY required")
	}
	_ = keyword // 预留：构建 SearchItems 请求体并 SigV4 签名后 POST
	return nil, fmt.Errorf("native PA-API not implemented, use Python tool via amazon analyze")
}
