import { DefaultTheme } from "vitepress"

type Sidebars = DefaultTheme.SidebarItem[]

interface ISides {
    lang: string
    list: Sidebars
}

export default function rewrites(froms: ISides, tos: ISides) {
    const from = froms.list.map(({ link }) => link)
    const to = tos.list.map(({ link }) => link)

    const rewrites: Record<string, string> = {}

    from.forEach((item, i) => {
        rewrites[`${froms.lang}/${item}.md`] = `${froms.lang}/${to[i]}.md`
    })

    to.forEach((item, i) => {
        rewrites[`${item}.md`] = `${from[i]}.md`
    })

    return rewrites
}
