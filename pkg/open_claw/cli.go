package open_claw

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/magao/agent/pkg/open_claw/tools"
)

// Execute runs the OpenClaw CLI (mirrors openclaw gateway, agent, onboard, doctor)
func Execute() {
	cfg, err := LoadConfig()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	root := &cobra.Command{
		Use:   "openclaw",
		Short: "OpenClaw — Personal AI Assistant (Go implementation)",
		Long: `OpenClaw is a personal AI assistant you run on your own devices.
It answers on channels you use (WhatsApp, Telegram, Slack, Discord, WebChat).
The Gateway is the control plane — ws://127.0.0.1:18789`,
	}

	// gateway
	gatewayCmd := &cobra.Command{
		Use:   "gateway",
		Short: "Start the Gateway (WebSocket control plane)",
		RunE: func(cmd *cobra.Command, args []string) error {
			port, _ := cmd.Flags().GetInt("port")
			if port > 0 {
				if cfg.Gateway == nil {
					cfg.Gateway = &GatewayConfig{}
				}
				cfg.Gateway.Port = port
			}
			verbose, _ := cmd.Flags().GetBool("verbose")
			if verbose {
				fmt.Println("[openclaw] starting gateway...")
			}
			store := NewSessionStore()
			agent := NewLLMAgent(cfg)
			gw := NewGateway(cfg, store, agent)
			return gw.Run()
		},
	}
	gatewayCmd.Flags().IntP("port", "p", 18789, "port")
	gatewayCmd.Flags().BoolP("verbose", "v", false, "verbose")

	// agent
	agentCmd := &cobra.Command{
		Use:   "agent",
		Short: "Send a message to the agent (requires gateway running)",
		RunE: func(cmd *cobra.Command, args []string) error {
			msg, _ := cmd.Flags().GetString("message")
			if msg == "" && len(args) > 0 {
				msg = args[0]
			}
			if msg == "" {
				return fmt.Errorf("--message required")
			}
			// 亚马逊产品分析：检测关键词并注入 PA-API 数据
			enrichedMsg := msg
			if shouldUseAmazonAnalysis(msg) {
				client := tools.NewAmazonClientFromEnv()
				kw := extractAmazonKeyword(msg)
				if kw != "" {
					if data, err := client.AnalyzeForProductDev(kw); err == nil {
						enrichedMsg = "【亚马逊商品数据】\n" + data + "\n【用户问题】" + msg
					}
				}
			}
			store := NewSessionStore()
			sess := NewSession("main", "default")
			store.Put(sess)
			agent := NewLLMAgent(cfg)
			reply, err := agent.Run(sess.Key, enrichedMsg)
			if err != nil {
				return err
			}
			fmt.Println(reply)
			return nil
		},
	}
	agentCmd.Flags().StringP("message", "m", "", "message to send")
	agentCmd.Flags().String("thinking", "", "thinking level (off|minimal|low|medium|high)")

	// onboard
	onboardCmd := &cobra.Command{
		Use:   "onboard",
		Short: "Run the onboarding wizard",
		RunE: func(cmd *cobra.Command, args []string) error {
			installDaemon, _ := cmd.Flags().GetBool("install-daemon")
			if installDaemon {
				fmt.Println("[openclaw] install-daemon: run `openclaw gateway` in a separate terminal or as a systemd/launchd service")
			}
			home := OpenClawHome()
			configPath := ConfigPath()
			fmt.Printf("OpenClaw home: %s\n", home)
			fmt.Printf("Config path: %s\n", configPath)
			if err := os.MkdirAll(home, 0755); err != nil {
				return err
			}
			workspace := cfg.Workspace
			if workspace == "" {
				workspace = "~/.openclaw/workspace"
			}
			wsDir := expandHome(workspace)
			if err := os.MkdirAll(wsDir, 0755); err != nil {
				return err
			}
			fmt.Printf("Workspace: %s\n", wsDir)
			fmt.Println("\nOnboarding complete. Next steps:")
			fmt.Println("  1. openclaw gateway --port 18789")
			fmt.Println("  2. openclaw agent --message 'Hello'")
			return nil
		},
	}
	onboardCmd.Flags().Bool("install-daemon", false, "install daemon (launchd/systemd)")

	// doctor
	doctorCmd := &cobra.Command{
		Use:   "doctor",
		Short: "Run diagnostics",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Println("OpenClaw Doctor")
			fmt.Println("---------------")
			home := OpenClawHome()
			configPath := ConfigPath()
			fmt.Printf("  Home: %s\n", home)
			if _, err := os.Stat(home); err != nil {
				fmt.Printf("  [WARN] Home dir missing: %v\n", err)
			} else {
				fmt.Printf("  [OK] Home dir exists\n")
			}
			fmt.Printf("  Config: %s\n", configPath)
			if _, err := os.Stat(configPath); err != nil {
				fmt.Printf("  [WARN] Config missing (using defaults)\n")
			} else {
				fmt.Printf("  [OK] Config exists\n")
			}
			if cfg.Agent != nil && cfg.Agent.Model != "" {
				fmt.Printf("  Model: %s\n", cfg.Agent.Model)
			}
			fmt.Println("---------------")
			return nil
		},
	}

	// message send
	messageCmd := &cobra.Command{
		Use:   "message",
		Short: "Message commands",
	}
	sendCmd := &cobra.Command{
		Use:   "send",
		Short: "Send a message to a channel",
		RunE: func(cmd *cobra.Command, args []string) error {
			to, _ := cmd.Flags().GetString("to")
			message, _ := cmd.Flags().GetString("message")
			if to == "" || message == "" {
				return fmt.Errorf("--to and --message required")
			}
			fmt.Printf("Send to %s: %s\n(requires gateway + channel configured)\n", to, message)
			return nil
		},
	}
	sendCmd.Flags().String("to", "", "recipient")
	sendCmd.Flags().StringP("message", "m", "", "message")
	messageCmd.AddCommand(sendCmd)

	// amazon analyze
	amazonCmd := &cobra.Command{
		Use:   "amazon",
		Short: "亚马逊商品分析（PA-API 5.0）",
	}
	analyzeCmd := &cobra.Command{
		Use:   "analyze",
		Short: "抓取亚马逊商品并分析，输出可开发投入市场的产品建议",
		RunE: func(cmd *cobra.Command, args []string) error {
			keyword, _ := cmd.Flags().GetString("keyword")
			if keyword == "" && len(args) > 0 {
				keyword = args[0]
			}
			if keyword == "" {
				return fmt.Errorf("--keyword 或传入关键词参数")
			}
			client := tools.NewAmazonClientFromEnv()
			report, err := client.AnalyzeForProductDev(keyword)
			if err != nil {
				return err
			}
			fmt.Println(report)
			return nil
		},
	}
	analyzeCmd.Flags().StringP("keyword", "k", "", "类目/产品关键词")
	amazonCmd.AddCommand(analyzeCmd)

	root.AddCommand(gatewayCmd, agentCmd, onboardCmd, doctorCmd, messageCmd, amazonCmd)

	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}

func shouldUseAmazonAnalysis(msg string) bool {
	s := strings.ToLower(msg)
	return strings.Contains(s, "亚马逊") || strings.Contains(s, "amazon") ||
		strings.Contains(s, "分析") && (strings.Contains(s, "市场") || strings.Contains(s, "产品") || strings.Contains(s, "商品"))
}

func extractAmazonKeyword(msg string) string {
	msg = strings.TrimSpace(msg)
	for _, prefix := range []string{"分析", "抓取", "搜索", "查询"} {
		if strings.HasPrefix(msg, prefix) {
			msg = strings.TrimSpace(strings.TrimPrefix(msg, prefix))
			break
		}
	}
	for _, suffix := range []string{"市场", "产品", "开发", "投入"} {
		msg = strings.TrimSuffix(msg, suffix)
		msg = strings.TrimSpace(msg)
	}
	if msg == "" {
		return "product"
	}
	return msg
}

func expandHome(p string) string {
	if len(p) < 2 || p[:2] != "~/" {
		return p
	}
	home, _ := os.UserHomeDir()
	if home == "" {
		return p
	}
	return home + p[1:]
}
