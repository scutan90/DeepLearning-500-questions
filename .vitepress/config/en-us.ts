import { type DefaultTheme, defineConfig } from "vitepress"

export default defineConfig({
    lang: "en-us",
    title: "Deep Learning 500 Questions",
    themeConfig: {

        sidebar: {
            "/en-us/": { base: "/en-us/", items: sidebarGuide() },
        },

        editLink: {
            pattern: "https://github.com/scutan90/DeepLearning-500-questions/edit/master/:path",
            text: "Edit this page on GitHub",
        },

        docFooter: {
            prev: "Last Page",
            next: "Next Page",
        },

        footer: {
            message: "Released under the MIT License.",
            copyright: "Copyright © 2018-present scutan90",
        },

        lastUpdated: {
            text: "Last updated at",
            formatOptions: {
                dateStyle: "short",
                timeStyle: "medium",
            },
        },
    },
})

export function sidebarGuide(): DefaultTheme.SidebarItem[] {
    return [
        { text: "guide", link: "README" },
        { text: "Chapter 1", link: "ch01_MathematicalBasis/MathematicalBasis" },
        { text: "Chapter 2", link: "ch02_MachineLearningFoundation/TheBasisOfMachineLearning" },
        { text: "Chapter 3", link: "ch03_DeepLearningFoundation/DeepLearningFoundation" },
        { text: "Chapter 4", link: "ch04_ClassicNetwork/ClassicNetwork" },
    ]
}
