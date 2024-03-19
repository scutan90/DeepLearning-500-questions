import { type DefaultTheme, defineConfig } from "vitepress"

export default defineConfig({
    lang: "zh-cn",
    themeConfig: {

        sidebar: {
            "/": { base: "/", items: sidebarGuide() },
        },

        editLink: {
            pattern: "https://github.com/scutan90/DeepLearning-500-questions/edit/master/:path",
            text: "在 GitHub 上编辑此页面",
        },

        docFooter: {
            prev: "上一页",
            next: "下一页",
        },

        footer: {
            message: "基于 MIT 许可发布",
            copyright: `版权所有 © 2018-${new Date().getFullYear()} scutan90`,
        },

        outline: {
            label: "页面导航",
        },

        lastUpdated: {
            text: "最后更新于",
            formatOptions: {
                dateStyle: "short",
                timeStyle: "medium",
            },
        },

        langMenuLabel: "多语言",
        returnToTopLabel: "回到顶部",
        sidebarMenuLabel: "菜单",
        darkModeSwitchLabel: "主题",
        lightModeSwitchTitle: "切换到浅色模式",
        darkModeSwitchTitle: "切换到深色模式",
    },
})

export function sidebarGuide(): DefaultTheme.SidebarItem[] {
    return [
        { text: "简介", link: "README" },
        { text: "第一章", link: "ch01_数学基础/第一章_数学基础" },
        { text: "第二章", link: "ch02_机器学习基础/第二章_机器学习基础" },
        { text: "第三章", link: "ch03_深度学习基础/第三章_深度学习基础" },
        { text: "第四章", link: "ch04_经典网络/第四章_经典网络" },
        { text: "第五章", link: "ch05_卷积神经网络(CNN)/第五章_卷积神经网络(CNN)" },
        { text: "第六章", link: "ch06_循环神经网络(RNN)/第六章_循环神经网络(RNN)" },
        { text: "第七章", link: "ch07_生成对抗网络(GAN)/ch7" },
        { text: "第八章", link: "ch08_目标检测/第八章_目标检测" },
        { text: "第九章", link: "ch09_图像分割/第九章_图像分割" },
        { text: "第十章", link: "ch10_强化学习/第十章_强化学习" },
        { text: "第十一章", link: "ch11_迁移学习/第十一章_迁移学习" },
        { text: "第十二章", link: "ch12_网络搭建及训练/第十二章_网络搭建及训练" },
        { text: "第十三章", link: "ch13_优化算法/第十三章_优化算法" },
        { text: "第十四章", link: "ch14_超参数调整/第十四章_超参数调整" },
        { text: "第十五章", link: "ch15_GPU和框架选型/第十五章_异构运算、GPU及框架选型" },
        { text: "第十六章", link: "ch16_自然语言处理(NLP)/第十六章_NLP" },
        { text: "第十七章", link: "ch17_模型压缩、加速及移动端部署/第十七章_模型压缩、加速及移动端部署" },
        { text: "第十八章", link: "ch18_后端架构选型、离线及实时计算/第十八章_后端架构选型、离线及实时计算" },
        { text: "第十八章", link: "ch18_后端架构选型及应用场景/第十八章_后端架构选型及应用场景" },
        { text: "第十九章", link: "ch19_软件专利申请及权利保护/第十九章_软件专利申请及权利保护" },
    ]
}


