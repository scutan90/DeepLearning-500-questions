import { defineConfig } from "vitepress"
import sidebar from "./sidebar"

// https://vitepress.dev/reference/site-config
export default defineConfig({
    title: "DeepLearning-500-questions",
    description: "Deep learning Q&A",
    themeConfig: {
        // https://vitepress.dev/reference/default-theme-config
        nav: [
            { text: "Home", link: "/" },
        ],

        sidebar,

        socialLinks: [
            { icon: "github", link: "https://github.com/scutan90/DeepLearning-500-questions" },
        ],
    },
    markdown: {
        math: true,
        linkify: false,
    },
    vite: {
        assetsInclude: ["**/*.jpg", "**/*.jpeg", "**/*.bmp", "**/*.JPEG"],
    },
})
