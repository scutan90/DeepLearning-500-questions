import { defineConfig } from "vitepress"
import sidebar from "./zh-cn"

export default defineConfig({
    title: "深度学习500问",

    lastUpdated: true,
    cleanUrls: true,
    metaChunk: true,

    markdown: {
        math: true,
    },
    ignoreDeadLinks: true,

    themeConfig: {
        socialLinks: [
            { icon: "github", link: "https://github.com/scutan90/DeepLearning-500-questions" },
        ],
    },
    head: [
        ["meta", { property: "og:locale", content: "zh" }],
    ],

    vite: {
        assetsInclude: ["**/*.jpg", "**/*.jpeg", "**/*.bmp", "**/*.JPEG"],
    },
})